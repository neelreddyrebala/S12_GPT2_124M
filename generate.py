import torch
import tiktoken

from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

enc = tiktoken.get_encoding("gpt2")

config = GPTConfig(block_size=256)
model = GPT(config)
state_dict = torch.load("checkpoints/final_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 100):
    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return enc.decode(idx[0].tolist())


if __name__ == "__main__":
    prompt = "Hello, my name is?"
    out = generate_text(prompt, max_new_tokens=80)
    print("\n=== SAMPLE GENERATION ===")
    print(out)
