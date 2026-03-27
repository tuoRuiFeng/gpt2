from datasets import load_dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset
    
def load_local_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 去掉空行
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    return lines
    
def get_tokenizer():
    return GPT2Tokenizer.from_pretrained("gpt2")

def tokenize(example, tokenizer, max_length=128):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )