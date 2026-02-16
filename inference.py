"""
Head-swap inference: put source face onto target body.
Pipeline: align -> generate head -> optional identity/SR -> mask -> paste onto target.
"""
from model.AlignModule.generator import FaceGenerator
from model.BlendModule.generator import Generator as Decoder
from model.AlignModule.config import Params as AlignParams
from model.BlendModule.config import Params as BlendParams 
from model.third.faceParsing.model import BiSeNet
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
import cv2
import numpy as np
from process.process_func import Process
from process.process_utils import *
import os
from datetime import datetime
import onnxruntime as ort
from utils.utils import color_transfer2


class Infer(Process):
    """Head-swap pipeline: face alignment, generation, optional identity/SR, then composite."""

    def __init__(self, align_path, blend_path, parsing_path, params_path, bfm_folder):
        Process.__init__(self, params_path, bfm_folder)
        align_params = AlignParams()
        blend_params = BlendParams()
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        # Face parsing: skin/hair/eyes etc. -> head mask
        self.parsing = BiSeNet(n_classes=19).to(self.device)
        # Generator: source identity + target pose -> draft face
        self.netG = FaceGenerator(align_params).to(self.device)
        # Decoder: refines draft with parsing masks
        self.decoder = Decoder(blend_params).to(self.device)
        self.loadModel(align_path, blend_path, parsing_path)
        self.eval_model(self.netG, self.decoder, self.parsing)
        # Optional super-resolution (ONNX)
        self.ort_session_sr = ort.InferenceSession('./pretrained_models/sr_cf.onnx', providers=['CPUExecutionProvider'])

    def _duplicate_source_identity(self, gen, src_align, src_lmk, tgt_lmk, alpha=0.95, debug_dir=None, step_name="05"):
        """Warp source face to target layout and blend onto gen (main identity step)."""
        H, W = gen.shape[:2]
        src_lmk = np.asarray(src_lmk, dtype=np.float32)
        tgt_lmk = np.asarray(tgt_lmk, dtype=np.float32)
        if src_lmk.ndim == 3:
            src_lmk = src_lmk.reshape(-1, src_lmk.shape[-1])
        if tgt_lmk.ndim == 3:
            tgt_lmk = tgt_lmk.reshape(-1, tgt_lmk.shape[-1])
        if src_lmk.shape[0] < 68 or tgt_lmk.shape[0] < 68:
            self._debug_log("  5.1. Identity duplicate: FAILED (insufficient landmarks)")
            return gen

        src_pts = src_lmk[:68, :2]
        tgt_pts = tgt_lmk[:68, :2]
        if np.any(np.isnan(src_pts)) or np.any(np.isnan(tgt_pts)):
            self._debug_log("  5.1. Identity duplicate: FAILED (NaN in landmarks)")
            return gen

        # Similarity transform: eyes, nose tip, mouth corners (5 anchors).
        anchor_ids = [36, 45, 30, 48, 54]
        src_anchor = src_pts[anchor_ids].astype(np.float32)
        tgt_anchor = tgt_pts[anchor_ids].astype(np.float32)
        M, inliers = cv2.estimateAffinePartial2D(src_anchor, tgt_anchor, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if M is None:
            self._debug_log("  5.1. Identity duplicate: FAILED (transform estimation failed)")
            return gen
        
        self._debug_log(f"  5.1. Similarity transform computed (inliers: {inliers.sum() if inliers is not None else 'N/A'}/5)")

        warped_src = cv2.warpAffine(
            src_align,
            M,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)
        gen_f = gen.astype(np.float32)
        
        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_a_warped_source.png", warped_src.astype(np.uint8))
            self._debug_log(f"  5.2. Source face warped to target geometry")

        # Clone mask: create minimal partial face mask (core features only) to avoid boundary/ear issues.
        # Use only inner face landmarks and restrict width to exclude ear regions.
        if tgt_pts.shape[0] >= 68:
            center_x = float((tgt_pts[36, 0] + tgt_pts[45, 0]) * 0.5)
            face_w = max(float(np.linalg.norm(tgt_pts[16] - tgt_pts[0])), 1.0)
            face_h = max(float(np.linalg.norm(tgt_pts[8] - ((tgt_pts[19] + tgt_pts[24]) * 0.5))), 1.0)
            
            # Create minimal mask using only inner face landmarks (exclude outer contour)
            # Use: eyebrows, eyes, nose, mouth - exclude jaw contour points that extend to ears
            inner_face_pts = np.concatenate([
                tgt_pts[17:27],  # Eyebrows
                tgt_pts[36:48],  # Eyes
                tgt_pts[27:36],  # Nose
                tgt_pts[48:68],  # Mouth and inner jaw
            ])
            
            # Create convex hull from inner points only
            inner_hull = cv2.convexHull(np.round(inner_face_pts).astype(np.int32))
            face_mask = np.zeros((H, W), dtype=np.float32)
            cv2.fillConvexPoly(face_mask, inner_hull, 1.0)
            
            # Restrict width: allow more expansion at top/bottom, less at sides (to exclude ears)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            brow_y = float(np.min(tgt_pts[17:27, 1]))
            chin_y = float(np.max(tgt_pts[6:12, 1]))
            mid_y = float((brow_y + chin_y) * 0.5)
            
            # Wider at top (forehead) and bottom (chin): 50% of face width
            # Narrower at middle (cheeks): 42% of face width (where ears are)
            top_bottom_width = 0.50 * face_w
            middle_width = 0.42 * face_w
            
            # Create adaptive width limit based on vertical position
            y_dist_from_mid = np.abs(yy - mid_y)
            y_range = max(chin_y - brow_y, 1.0)
            # Interpolate between top/bottom width and middle width
            width_factor = np.clip(1.0 - (y_dist_from_mid / (0.5 * y_range)), 0.0, 1.0)
            width_limit = middle_width + (top_bottom_width - middle_width) * width_factor
            
            side_exclusion = np.abs(xx - center_x) > width_limit
            face_mask[side_exclusion] = 0.0
            
            # Slight dilation to expand mask a bit more (but still conservative)
            face_mask_u8 = (face_mask > 0.5).astype(np.uint8)
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            face_mask_u8 = cv2.dilate(face_mask_u8, kernel_dilate, iterations=1)
            # Then slight erosion to smooth
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            face_mask_u8 = cv2.erode(face_mask_u8, kernel_erode, iterations=1)
            face_mask = face_mask_u8.astype(np.float32)
            
            # Keep only largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(face_mask_u8, connectivity=8)
            if num_labels > 1:
                largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                face_mask = (labels == largest).astype(np.float32)
            
            face_mask_base = face_mask.copy()
            
            # Smooth edges
            face_mask = cv2.GaussianBlur(face_mask, (9, 9), 0)
            face_mask = np.clip(face_mask, 0, 1.0)
            
            if debug_dir:
                self._debug_save(debug_dir, f"{step_name}_b_mask_base.png", face_mask_base, is_mask=True)
                self._debug_save(debug_dir, f"{step_name}_b_mask_blurred.png", face_mask, is_mask=True)
                self._debug_log(f"  5.3. Partial face mask created (inner landmarks, expanded, top/bottom: {top_bottom_width:.1f}px, middle: {middle_width:.1f}px from center)")
        else:
            # Fallback: simple convex hull
            hull = cv2.convexHull(np.round(tgt_pts).astype(np.int32))
            face_mask = np.zeros((H, W), dtype=np.float32)
            cv2.fillConvexPoly(face_mask, hull, 1.0)
            face_mask_base = face_mask.copy()
            face_mask = cv2.GaussianBlur(face_mask, (11, 11), 0)
            if debug_dir:
                self._debug_save(debug_dir, f"{step_name}_b_mask_base.png", face_mask_base, is_mask=True)
                self._debug_log(f"  5.3. Fallback: Simple landmark-based mask created")

        # Attenuate mask on lower/side contour to avoid hard seam (e.g. below ear).
        face_u8 = (face_mask > 0.5).astype(np.uint8)
        eroded = cv2.erode(face_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
        ring = np.clip(face_u8.astype(np.float32) - eroded.astype(np.float32), 0.0, 1.0)
        ring = cv2.GaussianBlur(ring, (13, 13), 0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        center_x = float((tgt_pts[36, 0] + tgt_pts[45, 0]) * 0.5)
        face_w = max(float(np.linalg.norm(tgt_pts[16] - tgt_pts[0])), 1.0)
        face_h = max(float(np.linalg.norm(tgt_pts[8] - ((tgt_pts[19] + tgt_pts[24]) * 0.5))), 1.0)
        lower_gate = np.clip((yy - (tgt_pts[30, 1] + 0.10 * face_h)) / (0.35 * face_h), 0.0, 1.0)
        side_gate = np.clip((np.abs(xx - center_x) - 0.30 * face_w) / (0.18 * face_w), 0.0, 1.0)
        right_gate = np.clip((xx - center_x) / (0.22 * face_w), 0.0, 1.0)
        atten = np.clip(ring * lower_gate * side_gate * (0.65 + 0.35 * right_gate), 0.0, 1.0)
        face_mask = np.clip(face_mask - 0.55 * atten, 0.0, 1.0)
        
        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_c_mask_attenuated.png", face_mask, is_mask=True)
            self._debug_save(debug_dir, f"{step_name}_c_attenuation_map.png", atten, is_mask=True)
            self._debug_log(f"  5.4. Mask attenuated at edges (face_w={face_w:.1f}, face_h={face_h:.1f})")

        # Forehead: blend clone with gen to match skin tone.
        brow = tgt_pts[17:27]
        brow_min_y = float(np.min(brow[:, 1]))
        brow_max_y = float(np.max(brow[:, 1]))
        y0 = int(np.clip(brow_min_y - 0.40 * face_h, 0, H - 1))
        y1 = int(np.clip(brow_max_y + 0.04 * face_h, 0, H - 1))
        x0 = int(np.clip(np.min(brow[:, 0]) - 0.10 * face_w, 0, W - 1))
        x1 = int(np.clip(np.max(brow[:, 0]) + 0.10 * face_w, 0, W - 1))
        if y1 > y0 and x1 > x0:
            fm = np.zeros((H, W), dtype=np.float32)
            cv2.rectangle(fm, (x0, y0), (x1, y1), 1.0, -1)
            fm = cv2.GaussianBlur(fm, (31, 31), 0)
            fm3 = np.clip(fm * 0.75 * face_mask, 0.0, 1.0)[:, :, np.newaxis]
            warped_src_before_forehead = warped_src.copy()
            warped_src = warped_src * (1.0 - fm3) + (0.45 * warped_src + 0.55 * gen_f) * fm3
            
            if debug_dir:
                self._debug_save(debug_dir, f"{step_name}_d_forehead_mask.png", fm, is_mask=True)
                self._debug_save(debug_dir, f"{step_name}_d_warped_before_forehead.png", warped_src_before_forehead.astype(np.uint8))
                self._debug_save(debug_dir, f"{step_name}_d_warped_after_forehead.png", warped_src.astype(np.uint8))
                self._debug_log(f"  5.5. Forehead blended (region: [{x0},{y0}] to [{x1},{y1}])")
        else:
            if debug_dir:
                self._debug_log(f"  5.5. Forehead blending skipped (invalid region)")

        # Color transfer: match warped source skin tone to generated face skin tone
        # This ensures the partial mask part matches the rest of the face
        face_mask_uint8 = (face_mask * 255).astype(np.uint8)
        # Ensure mask is 3D (H, W, 1) for color_transfer2
        if face_mask_uint8.ndim == 2:
            face_mask_uint8 = face_mask_uint8[:, :, np.newaxis]
        warped_src_color_matched = color_transfer2(gen_f, warped_src, center_ratio=0.7, mask=face_mask_uint8)
        warped_src = warped_src_color_matched.astype(np.float32)
        
        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_e_color_matched.png", warped_src.astype(np.uint8))
            self._debug_log(f"  5.5.5. Color transfer applied: source face matched to target skin tone")

        blend = np.clip(face_mask * alpha, 0.0, 1.0)[:, :, np.newaxis]

        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_e_final_blend_mask.png", blend[:, :, 0], is_mask=True)
            self._debug_log(f"  5.6. Final blend mask created (alpha={alpha}, mask strength: min={blend.min():.3f}, max={blend.max():.3f}, mean={blend.mean():.3f})")

        out = warped_src * blend + gen_f * (1.0 - blend)
        result = np.clip(out, 0, 255).astype(np.uint8)
        
        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_f_before_blend.png", gen_f.astype(np.uint8))
            self._debug_save(debug_dir, f"{step_name}_f_after_blend.png", result)
            self._debug_log(f"  5.7. Final blending complete: {alpha*100:.0f}% warped source + {(1-alpha)*100:.0f}% generated")
        
        return result

    def _debug_log(self, msg):
        """Print one pipeline step (prefix [HeadSwap])."""
        print("[HeadSwap] %s" % msg)

    def _debug_save(self, debug_dir, name, img, is_mask=False):
        """Write image or mask to debug_dir for step-by-step inspection."""
        if not debug_dir or not os.path.isdir(debug_dir):
            return
        path = os.path.join(debug_dir, name)
        if is_mask and img.ndim == 2:
            out = (np.clip(img, 0, 1).astype(np.float32) * 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        else:
            out = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        cv2.imwrite(path, out)

    def run(self, src_img_path_list, tgt_img_path_list, save_base, crop_align=False, cat=False, use_sr=True, use_identity_duplicate=True, debug_dir=None):
        """Batch: for each (src, tgt) run_single and save result under save_base."""
        os.makedirs(save_base,exist_ok=True)
        i = 0
        for src_img_path,tgt_img_path in zip(src_img_path_list,tgt_img_path_list):
            gen = self.run_single(
                src_img_path,
                tgt_img_path,
                crop_align=crop_align,
                cat=cat,
                use_sr=use_sr,
                use_identity_duplicate=use_identity_duplicate,
                debug_dir=debug_dir,
            )
            img_name = os.path.splitext(os.path.basename(src_img_path))[0]+'-' + \
                        os.path.splitext(os.path.basename(tgt_img_path))[0]+'.png'
            cv2.imwrite(os.path.join(save_base,img_name),gen)
            print('\rhave done %04d' % i, end='', flush=True)
            i += 1
        print()

    def run_single(self, src_img_path, tgt_img_path, crop_align=False, cat=False, use_sr=True, use_identity_duplicate=True, debug_dir=None):
        """One head swap: align -> generate -> optional identity/SR -> mask -> paste -> return full-res image."""
        src_name = os.path.splitext(os.path.basename(src_img_path))[0]
        tgt_name = os.path.splitext(os.path.basename(tgt_img_path))[0]
        step_dir = None
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            step_dir = os.path.join(debug_dir, "%s-%s-%s" % (src_name, tgt_name, timestamp))
            os.makedirs(step_dir, exist_ok=True)
            self._debug_log("Debug output directory: %s" % step_dir)

        tgt_img = cv2.imread(tgt_img_path)
        tgt_align = tgt_img.copy()
        
        # 1. Align target to 512x512; keep inverse transform for later paste-back.
        tgt_align,info = self.preprocess_align(tgt_img)
        if tgt_align is None:
            self._debug_log("1. Target align: FAILED (no face detected)")
            return None
        self._debug_log("1. Target aligned 512x512")
        if step_dir:
            self._debug_save(step_dir, "01_tgt_align.png", tgt_align)

        src_img = cv2.imread(src_img_path)
        if src_img is None:
            raise FileNotFoundError(f"Could not read source image: {src_img_path}")
        # 2. Align source to 512x512 if crop_align (needed for identity steps).
        src_align = src_img
        info_src = None
        if crop_align:
            src_align, info_src = self.preprocess_align(src_img, top_scale=0.5)
            if src_align is None:
                raise ValueError(
                    f"No face detected in source image: {src_img_path}. "
                    "Try another image or use crop_align=False."
                )
        self._debug_log("2. Source aligned 512x512")
        if step_dir:
            self._debug_save(step_dir, "02_src_align.png", src_align)

        src_inp = self.preprocess(src_align)
        tgt_inp = self.preprocess(tgt_align)

        # 3. Target pose/expression (3DMM) then generate swapped head (generator + decoder).
        tgt_params = self.get_params(cv2.resize(tgt_align, (256, 256)),
                                      info['rotated_lmk'] / 2.0).unsqueeze(0)
        gen = self.forward(src_inp, tgt_inp, tgt_params)
        gen = self.postprocess(gen[0])
        self._debug_log("3. Generated head (generator + decoder)")
        if step_dir:
            self._debug_save(step_dir, "03_gen.png", gen)

        # 4. Optional super-resolution.
        if use_sr:
            gen = self.run_sr(gen)
            self._debug_log("4. Super-resolution applied")
            if step_dir:
                self._debug_save(step_dir, "04_gen_sr.png", gen)
        else:
            self._debug_log("4. Super-resolution skipped (use_sr=False)")

        # 5. Optional: warp source face onto gen (strong identity).
        if use_identity_duplicate and crop_align and info_src is not None and 'rotated_lmk' in info_src and 'rotated_lmk' in info:
            self._debug_log("5. Identity duplicate: Starting sub-processes...")
            gen = self._duplicate_source_identity(gen, src_align, info_src['rotated_lmk'], info['rotated_lmk'], debug_dir=step_dir, step_name="05")
            self._debug_log("5. Identity duplicate: COMPLETE")
            if step_dir:
                self._debug_save(step_dir, "05_gen_identity_final.png", gen)
        else:
            self._debug_log("5. Identity duplicate skipped")

        # 6. Head mask at 512: parsing output, largest component only (no ghost islands).
        face_mask_512 = np.clip(self.mask, 0, 1.0).astype(np.float32)
        face_mask_u8 = (face_mask_512 > 0.5).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(face_mask_u8, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            face_mask_u8 = (labels == largest).astype(np.uint8)
        face_mask_512 = face_mask_u8.astype(np.float32)
        self._debug_log("6. Head mask (512) built, largest component only")
        if step_dir:
            self._debug_save(step_dir, "06_mask_512.png", face_mask_512, is_mask=True)
            self._debug_save(step_dir, "06_gen_512_final.png", gen)

        tgt_mask = info['mask'] if info['mask'].ndim == 3 else info['mask'][..., np.newaxis]
        tgt_mask = (tgt_mask.astype(np.uint8) * 255) if tgt_mask.max() <= 1 else tgt_mask.astype(np.uint8)
        if not use_identity_duplicate:
            gen = color_transfer2(tgt_align, gen, center_ratio=0.7, mask=tgt_mask)

        # 7. Warp head mask to full res: dilate, warp, largest component, neck feather.
        RotateMatrix = info['im'][:2]
        mask = (face_mask_512 * 255).astype(np.uint8)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
        mask = cv2.dilate(mask, kernel_dilate)
        mask = cv2.warpAffine(mask, RotateMatrix, (tgt_img.shape[1], tgt_img.shape[0]))
        mask = (mask.astype(np.float32) / 255.0).clip(0, 1)
        mask = (mask > 0.5).astype(np.float32)
        warped_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(warped_u8, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            warped_u8 = (labels == largest).astype(np.uint8)
        mask = warped_u8.astype(np.float32)
        mask_binary = mask.copy()

        h, w = mask.shape
        bottom_peak = int(mask.shape[0] * 0.46)
        feather_h = max(int(h * 0.34), 65)
        for y in range(h):
            if y > bottom_peak:
                t = (y - bottom_peak) / float(feather_h)
                t = np.clip(t, 0, 1)
                t = t * t * (3.0 - 2.0 * t)
                mask[y, :] *= (1.0 - t)
        mask = np.clip(mask, 0, 1.0)
        neck_start = max(0, bottom_peak - 15)
        neck_band = mask[neck_start:, :].copy()
        neck_band = cv2.GaussianBlur(neck_band, (21, 21), 0)
        mask[neck_start:, :] = np.clip(neck_band, 0, 1.0)
        head_solid = mask_binary.copy()
        head_solid[bottom_peak:, :] = 0.0
        mask = np.maximum(mask, head_solid)
        # Slight erosion on upper head only to reduce halo.
        upper = mask.copy()
        upper[bottom_peak:, :] = 0.0
        upper_u8 = (upper > 0.5).astype(np.uint8)
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        upper_u8 = cv2.erode(upper_u8, k_erode, iterations=1)
        mask[upper > 0] = upper_u8[upper > 0].astype(np.float32)
        mask = np.clip(mask, 0, 1.0)[:, :, np.newaxis]

        self._debug_log("7. Head mask warped and feathered to full resolution")
        # 8. Warp gen to full res and composite: head = gen, rest = original target.
        rotate_gen = cv2.warpAffine(gen, RotateMatrix, (tgt_img.shape[1], tgt_img.shape[0]))
        final = (rotate_gen * mask + tgt_img * (1 - mask)).astype(np.uint8)
        self._debug_log("8. Composited to full resolution -> final")
        if step_dir:
            self._debug_save(step_dir, "07_mask_full.png", mask[:, :, 0], is_mask=True)
            self._debug_save(step_dir, "08_final.png", final)
        return np.concatenate([tgt_img, final], 1) if cat else final
    
    def forward(self, xs, xt, params):
        """Generate swapped head: netG (source+target+pose) -> parsing -> decoder -> head mask."""
        with torch.no_grad():
            # Generator: source identity + target appearance + target pose -> draft face
            xg = F.adaptive_avg_pool2d(
                self.netG(F.adaptive_avg_pool2d(xs, 256), F.adaptive_avg_pool2d(xt, 256), params)['fake_image'],
                512,
            )
            # Parsing: segment draft and target (skin, hair, etc.)
            M_a = self.parsing(self.preprocess_parsing(xg))
            M_t = self.parsing(self.preprocess_parsing(xt))
            M_a = self.postprocess_parsing(M_a)
            M_t = self.postprocess_parsing(M_t)
            # Decoder: refine draft with masks
            xg_gray = TF.rgb_to_grayscale(xg, num_output_channels=1)
            fake = self.decoder(xg, xg_gray, xt, M_a, M_t, xt, train=False)
            # Build head mask from parsing (face + hair classes); store for run_single
            gen_mask = self.parsing(self.preprocess_parsing(fake))
            gen_mask = self.postprocess_parsing(gen_mask)
            gen_mask = gen_mask[0][0].cpu().numpy()
            mask_t = M_t[0][0].cpu().numpy()
            mask = np.zeros_like(gen_mask)
            for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18]:
                mask[gen_mask == i] = 1.0
                mask[mask_t == i] = 1.0
            self.mask = mask
        return fake
    
    def run_sr(self, input_np):
        """Upscale 512 face with ONNX super-resolution model."""
        input_np = cv2.cvtColor(input_np, cv2.COLOR_BGR2RGB)
        input_np = input_np.transpose((2, 0, 1))
        input_np = np.array(input_np[np.newaxis, :])
        outputs_onnx = self.ort_session_sr.run(None, {'input_image':input_np.astype(np.uint8)})

        out_put_onnx = outputs_onnx[0]
        outimg = out_put_onnx[0,...].transpose(1,2,0)
        outimg = cv2.cvtColor(outimg, cv2.COLOR_BGR2RGB)
        return outimg

        
    def loadModel(self, align_path, blend_path, parsing_path):
        ckpt = torch.load(align_path, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(ckpt['net_G_ema'])
        ckpt = torch.load(blend_path, map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(ckpt['G'], strict=False)
        self.parsing.load_state_dict(torch.load(parsing_path))

    def eval_model(self, *args):
        """Set all given modules to eval()."""
        for arg in args:
            arg.eval()


if __name__ == "__main__":
    model = Infer(
        'pretrained_models/epoch_00190_iteration_000400000_checkpoint.pt',
        'pretrained_models/Blender-401-00012900.pth',
        'pretrained_models/parsing.pth',
        'pretrained_models/epoch_20.pth',
        'pretrained_models/BFM'
    )
    src_paths = ['./assets/human.jpg']
    tgt_paths = ['assets/model.jpg']
    model.run(src_paths, tgt_paths, save_base='res-1125', crop_align=True, cat=False, debug_dir='res-1125/debug')