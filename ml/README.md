# ml

## Purpose
Training, export, and parity evaluation for models.

## Entrypoints
- `ml/model_interface.py` — model API
- `ml/trainers/train_linear.py` — example trainer
- `tools/bless_model_inference.py` — promotion helper

## Do-not-touch
- Deterministic seeds; record manifests per run; ONNX/native parity abs_err ≤ 1e-5
