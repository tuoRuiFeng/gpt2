import torch.nn as nn
import torch
from tqdm.notebook import tqdm
from tqdm import tqdm


def train(model, dataloader, optimizer, device, pad_token_id, epochs=50):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    batch_losses = []

    for epoch in tqdm(range(epochs), desc="Epoch"):
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)
            logits = model(tokens)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

    return batch_losses
