# HeadSwap Codebase Guide

This guide explains the main structure of the HeadSwap project so you can understand how head swapping works and where each part of the code lives.

---

## Top-Level Layout

| Item | Purpose |
|------|--------|
| **`inference.py`** | Main entry for **inference**: loads models and runs the full head-swap pipeline (align → generate → optional identity/SR → mask → paste). |
| **`train.py`** | Entry for **training**: parses args and launches either Align or Blend training. |
| **`process/`** | Face detection, alignment, cropping, and 3DMM params (preprocessing used by both inference and training). |
| **`model/`** | Neural networks: Align (PIRender-style), Blend (decoder), and third-party (face parsing, 3D recon). |
| **`dataloader/`** | PyTorch datasets and data loading for **training** (align and blend stages). |
| **`trainer/`** | Training loops, optimizers, and checkpoints for Align and Blend. |
| **`utils/`** | Shared helpers: data loader factory, color transfer, init, etc. |
| **`pretrained_models/`** | Checkpoints and assets (e.g. BFM, SR ONNX) for inference. |

---

## Folder Purposes

### `process/` — Preprocessing and Geometry

Handles everything before and after the neural networks: face detection, landmarks, cropping, and 3D face parameters.

- **`process_func.py`**  
  Defines **`Process`**, the base class used by **`Infer`** in `inference.py`:
  - **Face detection** (`FaceDetector`) and **2D/3D landmarks** (`face_alignment`).
  - **`preprocess_align`**: detect face → crop with padding → 512×512 aligned crop and `info` (landmarks, matrices, mask).
  - **`get_params`**: 3DMM (expression, pose, translation) via **Deep3dRec** for the target face.
  - **`preprocess`**: resize and normalize for model input.

- **`process_utils.py`**  
  - **`crop_with_padding`**: rotation from eye line, crop box, resize to 512, and build transform matrices `m` / `im` (crop ↔ original).
  - **`apply_transform`**: apply a 3×3 (or 2×3) matrix to landmarks.

- Other scripts (e.g. `crop_img.py`, `select_*.py`) are for data preparation or face selection, not the core inference path.

**Summary:** `process` = "get a 512×512 aligned face and 3DMM params from a raw image."

---

### `model/` — Neural Networks

All learnable models used for head swap and face parsing.

- **`AlignModule/`** (PIRender-style "align" stage)  
  - **`generator.py`**: **`FaceGenerator`** — mapping net, warping net, editing net. Takes source image + target image + **driving source** (3DMM params) and outputs a first draft swapped face.
  - **`lib/`**: Encoder, Mapping, Warping, Editing building blocks.
  - **`config.py`**: Params (channels, layers, etc.).
  - **`discriminator.py`**, **`criterion/`**: used only in **training** (AlignTrainer).

- **`BlendModule/`** (refine / "blend" stage)  
  - **`generator.py`**: **`Generator`** (Decoder in inference) — VGG feature extractor + decoder. Takes the draft face, its parsing mask, target face/mask, and refines the result.
  - **`module.py`**: VGG19 feature extractor and decoder modules.
  - **`config.py`**: Blend-model config.

- **`third/`** — Third-party or reused components  
  - **`faceParsing/`**: **BiSeNet** — semantic face parsing (skin, hair, eyes, etc.) used for the **head mask** and inside the blend model.
  - **`Deep3dRec/`**: 3D face recon — predicts 3DMM coefficients (expression, angles, translation) from a face image; used in **`Process.get_params`** to get the driving signal for the align generator.

**Summary:** `model` = "align generator + blend decoder + face parsing + 3DMM."

---

### `dataloader/` — Training Data

PyTorch datasets and loaders used only when **training** the model (align or blend stage).

- **`DataLoader.py`**: **`DatasetBase`** — base PyTorch `Dataset`.
- **`AlignLoader.py`**: **`AlignData`** — dataset for training the **Align** (face reenactment) stage.
- **`BlendLoader.py`**: **`BlendData`** — dataset for training the **Blend** (refinement) stage.
- **`augmentation.py`**: **`ParametricAugmenter`** — augmentations for training.

**Inference does not use these**; it reads images from disk and uses **process** for alignment.

---

### `trainer/` — Training Logic

Training loops, optimizers, and checkpointing for the two model stages.

- **`ModelTrainer.py`**: **`ModelTrainer`** — base training loop (train/val steps, logging, checkpointing).
- **`AlignTrainer.py`**: **`AlignTrainer`** — trains **`FaceGenerator`** (and discriminator) using align data and losses.
- **`BlendTrainer.py`**: **`BlendTrainer`** — trains the **Blend** generator (decoder) using blend data.

**Summary:** `trainer` = "how we train the two model stages." Not used at inference time.

---

### `utils/` — Shared Utilities

- **`utils.py`**:
  - **`get_data_loader`**: builds dataloaders for **training** (uses `AlignData` / `BlendData`).
  - **`color_transfer2`**: LAB color transfer (used in inference when not using identity duplicate).
  - **`setup_seed`**, **`merge_args`**, init/EMA helpers for training.
- **`visualizer.py`**: training visualization (e.g. TensorBoard-style).

**Summary:** `utils` = "data loading for training + color transfer and small helpers."

---

## How Inference Fits Together

1. **`inference.py`**  
   Builds **`Infer(Process)`**: loads Align generator, Blend decoder, BiSeNet, SR ONNX, and **Process** (detection, landmarks, 3DMM).

2. **Process (`process/`)**  
   Aligns target and (optionally) source to 512×512; gets 3DMM params for target.

3. **Model (`model/`)**  
   - **AlignModule**: `FaceGenerator(source, target, target_3dmm)` → draft face.  
   - **BlendModule**: Decoder + BiSeNet parsing → refined 512 face + head mask.  
   - **third**: BiSeNet for mask; Deep3dRec for 3DMM in Process.

4. **Post-processing in `inference.py`**  
   Optional SR; optional identity duplicate (warp source to gen/target layout, blend); mask building and feathering; paste onto full-resolution target image.

**In short:** **process** prepares crops and params, **model** does the actual head swap and mask, **inference.py** orchestrates and composites. **dataloader** and **trainer** are only for training the models you later load in inference.

---

## Quick Reference: Inference Pipeline Steps

| Step | Where | What |
|------|--------|------|
| 1. Target align | `process` | Detect face, landmarks, crop → 512×512, store inverse transform |
| 2. Source align | `process` | Same for source (if `crop_align=True`) |
| 3. Generate head | `model` (Align + Blend) | Source + target + 3DMM → draft → parsing → refined 512 face + mask |
| 4. Super-resolution | `inference.py` + ONNX | Optional upscale |
| 5. Identity duplicate | `inference.py` | Optional: warp source to gen/target layout, blend onto gen |
| 6–7. Identity boost / source eyes | `inference.py` | Optional extra identity or eye paste |
| 8. Head mask | `inference.py` | Parsing mask → dilate, warp to full res, neck feather |
| 9–10. Composite | `inference.py` | Warp 512 result to full res, paste with mask onto target image |

---

## Running Inference

```bash
python inference.py
```

Default in `inference.py` uses:

- Source: `./assets/human.jpg`
- Target: `assets/model.jpg`
- Output: `res-1125/` (final images) and optionally `res-1125/debug/` (step-by-step images and logs when `debug_dir` is set).

To customize, edit the `model.run(...)` call at the bottom of `inference.py` (paths, `crop_align`, `use_sr`, `use_identity_duplicate`, `debug_dir`, etc.).
