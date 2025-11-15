import os
import math

import torch
import tiktoken

from model import GPT, GPTConfig

# -------------------
# Basic setup
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# -------------------
# Load data
# -------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
tokens = torch.tensor(enc.encode(text), dtype=torch.long)
print(f"Loaded {len(tokens)} tokens from input.txt")

# -------------------
# Hyperparameters
# -------------------
batch_size    = 8       # B
block_size    = 256     # T
max_steps     = 9000    # total training iterations
eval_interval = 50

warmup_steps  = 200
max_lr        = 3e-4
min_lr        = 1e-5

target_loss   = 0.099999

def get_lr(step):
    """Cosine LR schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = max(0.0, min(1.0, decay_ratio))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * cosine

# -------------------
# Data loader
# -------------------
def get_batch():
    ix = torch.randint(0, len(tokens) - block_size - 1, (batch_size,))
    x = torch.stack([tokens[i : i + block_size] for i in ix])
    y = torch.stack([tokens[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)

# -------------------
# Model & optimizer
# -------------------
config = GPTConfig(block_size=block_size)
model = GPT(config).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

log_path = os.path.join("logs", "train_log.txt")
best_loss = float("inf")

print(f"Training for up to {max_steps} steps (B={batch_size}, T={block_size})...")
print(f"Target training loss: {target_loss}")

with open(log_path, "w", encoding="utf-8") as log_file:
    for step in range(1, max_steps + 1):
        model.train()
        x, y = get_batch()

        # update LR
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        best_loss = min(best_loss, loss_val)

        if step % eval_interval == 0 or step == 1:
            line = f"Step {step:5d}, Loss: {loss_val:.6f}, LR: {lr:.6f}"
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

        # save occasional checkpoints
        if step % 1000 == 0:
            ckpt_path = os.path.join("checkpoints", f"model_step_{step}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # early stop if target reached
        if loss_val < target_loss:
            line = (
                f"Target loss of {target_loss} achieved at step {step}! "
                f"Final Loss: {loss_val:.4f}"
            )
            print(line)
            log_file.write(line + "\n")
            log_file.flush()
            break

final_path = os.path.join("checkpoints", "final_model.pt")
torch.save(model.state_dict(), final_path)
print(f"Training complete. Best loss: {best_loss:.6f}")
print(f"Final model saved to {final_path}")
print(f"Logs written to {log_path}")
