# models/build_vocab.py
import os
import csv
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# change TRAIN_CSV_FILE to the path of your downloaded CSV file
TRAIN_CSV_FILE = "data/train.csv" 
VOCAB_SIZE = 16000

def build_tokenizer():
    print(f"Building Tokenizer from CSV file: {TRAIN_CSV_FILE}")
    
    if not os.path.exists(TRAIN_CSV_FILE):
        print("Error: File not found!")
        return

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE, 
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<sos>", "<eos>"],
        show_progress=True
    )

    def csv_iterator():
        with open(TRAIN_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) # skip the first line header (en, zh)
            
            for row in reader:
                if len(row) >= 2:
                    yield row[0] # Yield English text
                    yield row[1] # Yield Chinese text

    print("Training tokenizer...")
    tokenizer.train_from_iterator(csv_iterator(), trainer=trainer)
    
    os.makedirs("config", exist_ok=True)
    save_path = "config/tokenizer_small_en_zh.json"
    tokenizer.save(save_path)
    print(f"Success! Tokenizer saved to {save_path}")

if __name__ == '__main__':
    build_tokenizer()