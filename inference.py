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

    def _gauss_pyr(self, img, levels):
        """Gaussian pyramid: list of downsampled images."""
        G = [img.astype(np.float32)]
        for _ in range(levels):
            G.append(cv2.pyrDown(G[-1]))
        return G

    def _lap_pyr(self, G):
        """Laplacian pyramid from Gaussian pyramid."""
        L = []
        for i in range(len(G) - 1):
            GE = cv2.pyrUp(G[i + 1], dstsize=(G[i].shape[1], G[i].shape[0]))
            L.append(G[i] - GE)
        L.append(G[-1].copy())
        return L

    def _reconstruct_from_lap(self, L):
        """Reconstruct image from Laplacian pyramid."""
        img = L[-1].copy()
        for i in range(len(L) - 2, -1, -1):
            img = cv2.pyrUp(img, dstsize=(L[i].shape[1], L[i].shape[0])) + L[i]
        return np.clip(img, 0, 255).astype(np.uint8)

    def _multiband_blend(self, src, dst, mask, levels=6, mask_sigma=15):
        """
        Multiband (Laplacian pyramid) blend for seamless composite.
        src: face region (e.g. warped source), dst: base (e.g. gen), mask: 0..255 single channel.
        """
        src = np.clip(src, 0, 255).astype(np.float32)
        dst = np.clip(dst, 0, 255).astype(np.float32)
        m = (np.clip(mask, 0, 255).astype(np.float32) / 255.0)
        if m.ndim == 2:
            m3 = np.stack([m, m, m], axis=-1)
        else:
            m3 = m
        m3 = cv2.GaussianBlur(m3, (0, 0), mask_sigma)
        m3 = np.clip(m3, 0.0, 1.0)

        Gs = self._gauss_pyr(src, levels)
        Gd = self._gauss_pyr(dst, levels)
        Gm = self._gauss_pyr(m3, levels)
        Ls = self._lap_pyr(Gs)
        Ld = self._lap_pyr(Gd)

        LS = []
        for ls, ld, gm in zip(Ls, Ld, Gm):
            gm = gm if gm.ndim == 3 else np.stack([gm, gm, gm], axis=-1)
            LS.append(ls * gm + ld * (1.0 - gm))
        return self._reconstruct_from_lap(LS)

    def _ring_tone_equalize(self, img_bgr, face_mask, skin_region=None):
        """Match inside/outside tone in a narrow ellipse boundary band."""
        img = img_bgr.astype(np.uint8)
        m = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        if skin_region is None:
            skin = np.ones_like(m, dtype=np.uint8)
        else:
            skin = (skin_region > 0).astype(np.uint8)

        inside_ring = ((m > 0.56) & (m < 0.80) & (skin > 0))
        outside_ring = ((m > 0.20) & (m < 0.44) & (skin > 0))
        band = ((m > 0.20) & (m < 0.80) & (skin > 0)).astype(np.float32)
        if int(np.count_nonzero(inside_ring)) < 100 or int(np.count_nonzero(outside_ring)) < 100:
            return img

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        k = max((int(0.06 * min(img.shape[:2])) | 1), 17)
        l_low = cv2.GaussianBlur(l, (k, k), 0)
        a_low = cv2.GaussianBlur(a, (k, k), 0)
        b_low = cv2.GaussianBlur(b, (k, k), 0)
        l_hi = l - l_low
        a_hi = a - a_low
        b_hi = b - b_low

        def _med(arr, idx):
            return float(np.median(arr[idx]))

        dl = np.clip(_med(l_low, outside_ring) - _med(l_low, inside_ring), -10.0, 10.0)
        da = np.clip(_med(a_low, outside_ring) - _med(a_low, inside_ring), -7.0, 7.0)
        db = np.clip(_med(b_low, outside_ring) - _med(b_low, inside_ring), -7.0, 7.0)

        w = cv2.GaussianBlur(band, (k, k), 0)
        w = np.clip(w * np.clip((m - 0.45) / 0.35, 0.0, 1.0), 0.0, 1.0)
        l_new = l_low + dl * w + l_hi
        a_new = a_low + da * w + a_hi
        b_new = b_low + db * w + b_hi
        out = cv2.merge([
            np.clip(l_new, 0, 255).astype(np.float32),
            np.clip(a_new, 0, 255).astype(np.float32),
            np.clip(b_new, 0, 255).astype(np.float32),
        ])
        return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _ring_border_blur(self, img_bgr, face_mask):
        """Strong blur on ellipse boundary band to hide contour line (essential for natural look)."""
        img = img_bgr.astype(np.float32)
        m = np.clip(face_mask.astype(np.float32), 0.0, 1.0)
        h, w = img.shape[:2]
        # Wider ring: full transition band so blur covers the visible seam
        band_half = 0.28
        ring = np.clip(1.0 - np.abs(m - 0.5) / band_half, 0.0, 1.0)
        ring *= ((m > 0.08) & (m < 0.92)).astype(np.float32)
        if int(np.count_nonzero(ring > 0.05)) < 80:
            return img_bgr.astype(np.uint8), ring

        # Strong blur: large kernel so border line is dissolved
        k1 = max((int(0.10 * min(h, w)) | 1), 35)
        k2 = max((int(0.06 * min(h, w)) | 1), 21)
        blur = cv2.GaussianBlur(img, (k1, k1), 0)
        blur = cv2.GaussianBlur(blur, (k2, k2), 0)
        ring_soft = cv2.GaussianBlur(ring, (k2, k2), 0)
        # High blend weight so blurred version dominates on border (hide line)
        w = np.clip(ring_soft * 1.0, 0.0, 0.95)[:, :, np.newaxis]
        out = np.clip(img * (1.0 - w) + blur * w, 0, 255).astype(np.uint8)
        return out, w[:, :, 0]

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

        # Clone mask: create ellipse-shaped partial face mask (core features only) to avoid boundary/ear issues.
        if tgt_pts.shape[0] >= 68:
            center_x = float((tgt_pts[36, 0] + tgt_pts[45, 0]) * 0.5)
            center_y = float((tgt_pts[36, 1] + tgt_pts[45, 1]) * 0.5)
            face_w = max(float(np.linalg.norm(tgt_pts[16] - tgt_pts[0])), 1.0)
            face_h = max(float(np.linalg.norm(tgt_pts[8] - ((tgt_pts[19] + tgt_pts[24]) * 0.5))), 1.0)
            
            # Calculate ellipse parameters based on face landmarks
            brow_y = float(np.min(tgt_pts[17:27, 1]))
            chin_y = float(np.max(tgt_pts[6:12, 1]))
            
            # Ellipse center: move down a little bit from eye center
            # Shift center_y down by ~8% of face height
            center_y_offset = face_h * 0.08
            center_y_adjusted = center_y + center_y_offset
            ellipse_center = (int(center_x), int(center_y_adjusted))
            
            # Ellipse axes: wider at top/bottom, narrower in middle (to exclude ears)
            # Much wider widths for bigger, fatter ellipse: Top/bottom width: 75% of face width, middle width: 65% of face width
            top_bottom_width = 0.8 * face_w
            middle_width = 0.7 * face_w
            
            # Ellipse height: from eyebrows to chin, with significant expansion
            ellipse_height = max(chin_y - brow_y, face_h * 0.8) * 1.20  # 30% taller
            
            # Create ellipse mask with adaptive width
            face_mask = np.zeros((H, W), dtype=np.float32)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            
            # Calculate distance from adjusted ellipse center
            dx = (xx - center_x) / (top_bottom_width / 2.0)
            dy = (yy - center_y_adjusted) / (ellipse_height / 2.0)
            
            # Adaptive width: wider at top/bottom, narrower in middle
            y_dist_from_center = np.abs(yy - center_y_adjusted)
            y_range = ellipse_height / 2.0
            width_factor = np.clip(1.0 - (y_dist_from_center / y_range), 0.0, 1.0)
            # Interpolate between top/bottom width and middle width
            adaptive_width = middle_width + (top_bottom_width - middle_width) * width_factor
            
            # Create ellipse with adaptive width
            dx_adaptive = (xx - center_x) / (adaptive_width / 2.0)
            ellipse_mask = (dx_adaptive * dx_adaptive + dy * dy) <= 1.0
            face_mask[ellipse_mask] = 1.0
            
            # Smooth the ellipse edges
            face_mask = cv2.GaussianBlur(face_mask, (11, 11), 0)
            face_mask = np.clip(face_mask, 0, 1.0)
            
            face_mask_base = face_mask.copy()
            
            if debug_dir:
                self._debug_save(debug_dir, f"{step_name}_b_mask_base.png", face_mask_base, is_mask=True)
                self._debug_save(debug_dir, f"{step_name}_b_mask_blurred.png", face_mask, is_mask=True)
                self._debug_log(f"  5.3. Ellipse-shaped partial face mask created (top/bottom: {top_bottom_width:.1f}px, middle: {middle_width:.1f}px, height: {ellipse_height:.1f}px)")
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

        # Keep source face details clear: skip additional color remap here.
        if debug_dir:
            self._debug_log("  5.5.5. Color transfer skipped (clarity-preserving mode)")

        blend = np.clip(face_mask * alpha, 0.0, 1.0)[:, :, np.newaxis]

        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_e_final_blend_mask.png", blend[:, :, 0], is_mask=True)
            self._debug_log(f"  5.6. Final blend mask created (alpha={alpha}, mask strength: min={blend.min():.3f}, max={blend.max():.3f}, mean={blend.mean():.3f})")

        # Multiband (Laplacian pyramid) composite for seamless ellipse boundary.
        mask_u8 = (np.clip(blend[:, :, 0], 0.0, 1.0) * 255.0).astype(np.uint8)
        mask_sigma = max(15, int(0.03 * min(H, W)))
        result = self._multiband_blend(
            np.clip(warped_src, 0, 255).astype(np.uint8),
            np.clip(gen_f, 0, 255).astype(np.uint8),
            mask_u8,
            levels=6,
            mask_sigma=mask_sigma,
        )

        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_f_before_blend.png", gen_f.astype(np.uint8))
            self._debug_save(debug_dir, f"{step_name}_f_after_blend.png", result)
            self._debug_log(f"  5.7. Multiband blend complete (levels=6, mask_sigma={mask_sigma})")

        # 5.8 One-skin harmonization:
        # Use inside-ellipse core skin as reference, then smoothly propagate it over full
        # face+neck skin (with boundary emphasis) to remove ellipse-region differences.
        try:
            h_res, w_res = result.shape[:2]
            res_t = self.preprocess(result, size=512)
            with torch.no_grad():
                parsing_out = self.parsing(self.preprocess_parsing(res_t))
            parsing_map = self.postprocess_parsing(parsing_out)[0, 0].cpu().numpy().astype(np.uint8)
            if parsing_map.shape != (h_res, w_res):
                parsing_map = cv2.resize(parsing_map, (w_res, h_res), interpolation=cv2.INTER_NEAREST)

            # Parsing skin/neck.
            skin_parse = ((parsing_map == 1) | (parsing_map == 14) | (parsing_map == 15)).astype(np.uint8)
            skin_parse = cv2.morphologyEx(
                skin_parse, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
            )
            skin_parse = cv2.dilate(
                skin_parse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
            )

            # Geometry face+neck support to avoid parsing holes.
            geom_region = np.zeros((h_res, w_res), dtype=np.uint8)
            hull = cv2.convexHull(np.round(tgt_pts[:27]).astype(np.int32))
            cv2.fillConvexPoly(geom_region, hull, 1)
            chin_y = float(np.max(tgt_pts[6:12, 1]))
            neck_bottom = int(np.clip(chin_y + face_h * 0.62, 0, h_res - 1))
            jaw_x0 = int(np.clip(np.min(tgt_pts[4:13, 0]) - 0.10 * face_w, 0, w_res - 1))
            jaw_x1 = int(np.clip(np.max(tgt_pts[4:13, 0]) + 0.10 * face_w, 0, w_res - 1))
            chin_top = int(np.clip(chin_y, 0, h_res - 1))
            if neck_bottom > chin_top and jaw_x1 > jaw_x0:
                cv2.rectangle(geom_region, (jaw_x0, chin_top), (jaw_x1, neck_bottom), 1, -1)
            geom_region = cv2.GaussianBlur(geom_region.astype(np.float32), (19, 19), 0)
            geom_region = (geom_region > 0.25).astype(np.uint8)

            target_region = ((skin_parse > 0) | (geom_region > 0)).astype(np.uint8)
            target_region = cv2.morphologyEx(
                target_region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1
            )

            inside = (face_mask > 0.50).astype(np.uint8)
            outside = (1 - inside).astype(np.uint8)
            core_ref = ((face_mask > 0.68).astype(np.uint8) & (target_region > 0)).astype(np.uint8)
            rest_skin = ((target_region > 0) & (core_ref == 0)).astype(np.uint8)

            if int(np.count_nonzero(core_ref)) > 180 and int(np.count_nonzero(rest_skin)) > 260:
                dist_in = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
                dist_out = cv2.distanceTransform(outside, cv2.DIST_L2, 5).astype(np.float32)
                signed_d = dist_in - dist_out

                # Build smooth correction field over target skin:
                # - stronger outside ellipse,
                # - gentle inside support,
                # - explicit boost near boundary to hide contour.
                outside_w = np.clip((1.0 - face_mask) * target_region.astype(np.float32), 0.0, 1.0)
                inside_w = np.clip((face_mask - 0.25) / 0.75, 0.0, 1.0) * target_region.astype(np.float32) * 0.35
                seam_sigma = max(10.0, 0.07 * min(face_w, face_h))
                seam_w = np.exp(-np.abs(signed_d) / seam_sigma) * target_region.astype(np.float32)
                corr_w = np.clip(0.78 * outside_w + 0.22 * inside_w + 0.55 * seam_w, 0.0, 1.0).astype(np.float32)
                corr_w = cv2.GaussianBlur(corr_w, (41, 41), 0)
                corr_w = np.clip(corr_w, 0.0, 1.0)

                lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
                l_ch, a_ch, b_ch = cv2.split(lab)
                k = max((int(min(h_res, w_res) * 0.12) | 1), 25)
                l_low = cv2.GaussianBlur(l_ch, (k, k), 0)
                a_low = cv2.GaussianBlur(a_ch, (k, k), 0)
                b_low = cv2.GaussianBlur(b_ch, (k, k), 0)
                l_high = l_ch - l_low
                a_high = a_ch - a_low
                b_high = b_ch - b_low

                def _mask_stats(arr, idx):
                    vals = arr[idx]
                    return float(np.median(vals)), float(np.std(vals))

                l_ref_m, l_ref_s = _mask_stats(l_low, core_ref > 0)
                a_ref_m, a_ref_s = _mask_stats(a_low, core_ref > 0)
                b_ref_m, b_ref_s = _mask_stats(b_low, core_ref > 0)
                l_rest_m, l_rest_s = _mask_stats(l_low, rest_skin > 0)
                a_rest_m, a_rest_s = _mask_stats(a_low, rest_skin > 0)
                b_rest_m, b_rest_s = _mask_stats(b_low, rest_skin > 0)

                l_scale = np.clip(l_ref_s / max(l_rest_s, 1e-6), 0.88, 1.14)
                a_scale = np.clip(a_ref_s / max(a_rest_s, 1e-6), 0.90, 1.12)
                b_scale = np.clip(b_ref_s / max(b_rest_s, 1e-6), 0.90, 1.12)

                l_target = (l_low - l_rest_m) * l_scale + l_rest_m + np.clip(l_ref_m - l_rest_m, -10.0, 10.0)
                a_target = (a_low - a_rest_m) * a_scale + a_rest_m + np.clip(a_ref_m - a_rest_m, -8.0, 8.0)
                b_target = (b_low - b_rest_m) * b_scale + b_rest_m + np.clip(b_ref_m - b_rest_m, -8.0, 8.0)

                strength = 0.95
                l_low_adj = l_low + (l_target - l_low) * corr_w * strength
                a_low_adj = a_low + (a_target - a_low) * corr_w * strength
                b_low_adj = b_low + (b_target - b_low) * corr_w * strength

                l_new = l_low_adj + l_high
                a_new = a_low_adj + a_high
                b_new = b_low_adj + b_high

                skin_soft = cv2.GaussianBlur(target_region.astype(np.float32), (35, 35), 0)
                skin_soft = np.clip(skin_soft, 0.0, 1.0)
                l_final = l_ch * (1.0 - skin_soft) + l_new * skin_soft
                a_final = a_ch * (1.0 - skin_soft) + a_new * skin_soft
                b_final = b_ch * (1.0 - skin_soft) + b_new * skin_soft

                lab_out = cv2.merge([
                    np.clip(l_final, 0, 255).astype(np.float32),
                    np.clip(a_final, 0, 255).astype(np.float32),
                    np.clip(b_final, 0, 255).astype(np.float32),
                ])
                result = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)

                # 5.8.1 Equalize ring tone (inside/outside ellipse skin).
                result = self._ring_tone_equalize(result, face_mask, target_region)

                if debug_dir:
                    self._debug_save(debug_dir, f"{step_name}_g_skin_parse.png", skin_parse.astype(np.float32), is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_geom_region.png", geom_region.astype(np.float32), is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_target_region.png", target_region.astype(np.float32), is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_core_ref.png", core_ref.astype(np.float32), is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_rest_skin.png", rest_skin.astype(np.float32), is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_corr_weight.png", corr_w, is_mask=True)
                    self._debug_save(debug_dir, f"{step_name}_g_unified_skin.png", result)
                    self._debug_log(
                        f"  5.8. One-skin harmonization applied (strength={strength:.2f})"
                    )
            elif debug_dir:
                self._debug_log("  5.8. One-skin harmonization skipped (insufficient reference/rest pixels)")
        except Exception as e:
            if debug_dir:
                self._debug_log(f"  5.8. One-skin harmonization skipped: {e}")

        # Always apply strong ellipse-border blur to hide contour (essential).
        result, ring_blur_w = self._ring_border_blur(result, face_mask)
        if debug_dir:
            self._debug_save(debug_dir, f"{step_name}_g_ring_blur_weight.png", ring_blur_w, is_mask=True)
            self._debug_save(debug_dir, f"{step_name}_final_border_blur.png", result)
            self._debug_log("  5.9. Strong border blur applied (ellipse contour hidden)")
        
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
    tgt_paths = ['assets/model1.jpg']
    model.run(src_paths, tgt_paths, save_base='res-1125', crop_align=True, cat=False, debug_dir='res-1125/debug')