# Transformer En-Zh Translation

**Project purpose**

A PyTorch implementation of a Transformer-based English-to-Chinese translation model tailored for the IWSLT 2017 En-Zh dataset. It includes training, validation, beam-search inference and BLEU evaluation.

---

## Quick start

1. Create a python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Edit hyperparameters and paths in `config/hyperparameters.json`. Important fields:
- `data_path_train`, `data_path_valid` – CSV files with `en,zh` columns
- `tokenizer_path` – path to SentencePiece / tokenizer json
- `save_path` – where the checkpoint will be saved
- `vocab_size` – shared vocabulary size used for training/SPM (default: 8000)
- `spm_vocab_size` – SPM training vocab size (default: 16000)
- `max_len` – max token length for source/target (default: 128)
- `batch_size`, `max_epochs`, `learning_rate`, etc.
- `beam_width`, `beam_alpha` – beam-search decoding parameters (defaults added to config)

> Note: This project keeps some general-purpose configuration fields that might belong to other experiments (e.g., `block_size`, `stride_overlap_ratio` used by other scripts). They are harmless and preserved for compatibility.

3. Train + Evaluate (single command):

```bash
python official_translation.py
```

- The script reads `config/hyperparameters.json`. Edit that file before running.
- Checkpoints are saved to the `save_path` in the config.

4. Run inference / compute BLEU

- The training script computes final BLEU on a test subset after training and prints random sample translations.
- Beam-search parameters are pulled from the config (`beam_width`, `beam_alpha`), so you can tune them without editing code.

---
## Tips & Troubleshooting

- If you want to evaluate an **existing** model only, set up your checkpoint at `save_path` and re-run `official_translation.py` (it will load and resume/evaluate).
- If you see training instability, try lowering `learning_rate` or increasing `warmup` in the config.
- To re-generate the tokenizer/training SPM, run `models/build_vocab.py` (edit path and `VOCAB_SIZE` in that file as needed).