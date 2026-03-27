import torch
from transformers import GPT2Tokenizer
from config import Config
from model import DemoTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# 导入模型
def load_model(model_path):
    cfg = Config()
    model = DemoTransformer(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=50):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)  # 逻辑得分

        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 采样

        tokens = torch.cat([tokens, next_token], dim=1)

        # 如果生成结束符可以提前停止
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens[0])


def main():
    model_path = "results/model.pt"

    tokenizer = GPT2Tokenizer.from_pretrained(
        "./tokenizer",
        local_files_only=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_path)

    # 固定输入内容（方便Jupyter使用，pycharm可改为终端输入）
    prompt = "Edward rose early on the New-year morning. He looked in every room and wished a"

    output = generate(model, tokenizer, prompt)
    print("\n生成结果：")
    print(output)
    print("-" * 50)


if __name__ == "__main__":
    main()
