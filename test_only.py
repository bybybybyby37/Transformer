# test_only.py
import torch
import torch.nn as nn
import os
import json
import sacrebleu
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset

from models import model as my_model
from models.data_interface import IWSLT17EnZhDataset, load_or_train_spm_for_iwslt17, collate_translation_batch
from official_translation import evaluate, beam_search_translate

def main():

    CONFIG_PATH = os.path.join("config", "hyperparameters.json")
    
    # choose model here
    MODEL_PATH = "checkpoints/best_model_synth.pt"
    
    # sample setting here, "None" if choose the full test set
    TEST_SAMPLES = None

    OUTPUT_FILE = "final_predictions_synth.txt"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    print(f"Testing Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Results will be saved to: {OUTPUT_FILE}")

    # preparing test dataset
    sp = load_or_train_spm_for_iwslt17(vocab_size=getattr(cfg, 'vocab_size', 8000))
    PAD_IDX = sp.pad_id()
    VOCAB_SIZE = sp.GetPieceSize()

    print("[Data] Loading IWSLT2017 Test Set...")
    dataset_dict = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-zh", trust_remote_code=True)
    test_set = IWSLT17EnZhDataset(dataset_dict["test"], sp, cfg.max_len, cfg.max_len, "en", "zh")
    
    collate_fn = lambda batch: collate_translation_batch(batch, pad_id=PAD_IDX)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # init model
    transformer = my_model.TransformerSeq2Seq(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, dropout=cfg.dropout,
        pad_idx=PAD_IDX
    ).to(DEVICE)

    # weight
    if os.path.exists(MODEL_PATH):
        print(f"Loading checkpoint from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            transformer.load_state_dict(checkpoint['model_state_dict'])
        else:
            transformer.load_state_dict(checkpoint)
        print("Weights loaded successfully.")
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    # PPL calculation
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.0) # without label_smoothing
    test_loss, test_ppl = evaluate(transformer, test_loader, criterion, VOCAB_SIZE, DEVICE)
    print(f"Test Set Results: Loss = {test_loss:.4f} | PPL = {test_ppl:.2f}")

    # BLEU calculation
    all_preds = []
    all_refs = []
    print("Calculating BLEU score...")
    
    # adjust here to generate long/short sentence
    BEAM_WIDTH = getattr(cfg, 'beam_width', 10)
    BEAM_ALPHA = getattr(cfg, 'beam_alpha', 1.2) 
    
    BEAM_WIDTH = 10
    BEAM_ALPHA = 1.1

    # auto decide test sample length
    if TEST_SAMPLES:
        indices = range(min(TEST_SAMPLES, len(test_loader.dataset)))
        print(f"Fast Mode: Testing first {len(indices)} samples.")
    else:
        indices = range(len(test_loader.dataset))
        print("Full Mode: Testing all samples.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
            
            for i in tqdm(indices, desc="Translating", unit="sent"):
                raw_item = test_loader.dataset.data[i]['translation']
                src_text = raw_item['en']
                tgt_text = raw_item['zh']
                
                with torch.no_grad():
                    pred_text = beam_search_translate(transformer, src_text, sp, DEVICE, cfg.max_len, beam_width=BEAM_WIDTH, alpha=BEAM_ALPHA)
                
                all_preds.append(pred_text)
                all_refs.append(tgt_text)

                # output will be write here
                f_out.write(f"Sample {i}:\n")
                f_out.write(f"Source: {src_text}\n")
                f_out.write(f"Ref:    {tgt_text}\n")
                f_out.write(f"Pred:   {pred_text}\n")
                f_out.write("-" * 50 + "\n") # 分隔符

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs], tokenize='zh')


    print(f"\n=========================================")
    print(f"TEST BLEU: {bleu.score:.2f}")
    print(f"Signature: {bleu}")
    print(f"=========================================")

    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs], remove_whitespace=True)
    
    print("=========================================")
    print(f"TEST chrF: {chrf.score:.2f}")
    print(f"Signature (chrF): {chrf}")
    print(f"=========================================")

if __name__ == "__main__":
    main()