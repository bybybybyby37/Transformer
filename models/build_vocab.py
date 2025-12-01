# models/build_vocab.py
import os
import csv  # 新增：引入 csv 模块
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# 修改路径为你实际下载的 csv 文件路径
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

    # 修改：适配 CSV 的迭代器
    def csv_iterator():
        with open(TRAIN_CSV_FILE, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None) # 跳过第一行标题 (en, zh)
            
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