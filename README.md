# ERA Session 12 — GPT-124M Decoder-Only Model (Assignment)

This project trains a GPT-2-small–style decoder-only model (~124M parameters) on `input.txt`
using a warmup + cosine learning rate schedule. The training loop stops early once the
**training loss** drops below **0.099999**, matching the assignment requirement.

**Target loss of 0.099999 achieved at step 6034! Final Loss: 0.0883
**

## Files

- `input.txt` — training corpus (provided by ERA; already included here).
- `model.py` — GPT model definition (decoder-only transformer).
- `train.py` — training loop with LR schedule and early stopping on target loss.
- `generate.py` — script to generate sample text from the trained model.
- `hf_space/app.py` — Gradio app for Hugging Face Spaces.
- `hf_space/requirements.txt` — Python dependencies for Spaces.
- `logs/` — training logs (created after running `train.py`).
- `checkpoints/` — model checkpoints (created after training).

## Basic Usage on Colab

1. Upload the zip and unzip:

   ```python
   from google.colab import files
   uploaded = files.upload()  # select the zip file
   !unzip -o s12_gpt_assignment_final.zip -d /content
   %cd /content/s12_gpt_assignment_final
   ```

2. Install dependencies:

   ```python
   !pip install torch tiktoken gradio
   ```

3. Train:

   ```python
   !python train.py
   ```

   - Prints `Step N, Loss: ..., LR: ...`
   - Stops early once `loss < 0.099999`
   - Writes logs to `logs/train_log.txt`
   - Saves final model to `checkpoints/final_model.pt`

4. Generate samples:

   ```python
   !python generate.py
   ```

## Hugging Face Spaces

1. Create a new Space (Gradio, Python).
2. Upload:
   - `model.py`
   - `checkpoints/final_model.pt` renamed/moved as `final_model.pt` at root
   - `hf_space/app.py` as `app.py` at root
   - `hf_space/requirements.txt` as `requirements.txt` at root
3. The Space will load `final_model.pt` and serve the Gradio UI.
