import torch
from torch.utils.data import DataLoader, Dataset
import os
import matplotlib.pyplot as plt
from config import Config
from model import DemoTransformer
from utils import load_local_data
from train import train

from transformers import GPT2Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===== 新增：Dataset类 =====
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx])
        }


def main():
    print("Hello world!")  # 测试开始运行
    cfg = Config()  # 超参数
    model = DemoTransformer(cfg).to(device)

    # ======= 加载本地数据 =======
    train_lines = load_local_data("data/wiki.train.tokens")
    valid_lines = load_local_data("data/wiki.valid.tokens")

    # ===== tokenizer =====
    tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
    tokenizer.pad_token = tokenizer.eos_token

    train_encodings = tokenizer(
        train_lines,
        truncation=True,
        padding="max_length",
        max_length=64  # 防止爆显存
    )

    valid_encodings = tokenizer(
        valid_lines,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    # ===== Dataset =====
    train_dataset = TextDataset(train_encodings)
    valid_dataset = TextDataset(valid_encodings)

    # ===== DataLoader =====
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8)

    # ===== 优化器 =====
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ===== 训练 =====
    losses = train(
        model,
        train_loader,
        optimizer,
        device,
        tokenizer.pad_token_id
    )
    os.makedirs("results", exist_ok=True)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    print("训练结束")

    # 保存结果
    plt.savefig("results/loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
