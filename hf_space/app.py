import gradio as gr
import torch
import tiktoken

from model import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

enc = tiktoken.get_encoding("gpt2")

config = GPTConfig(block_size=256)
model = GPT(config)
state_dict = torch.load("final_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


@torch.no_grad()
def generate_fn(prompt: str, max_new_tokens: int = 80):
    if not prompt.strip():
        return ""

    idx = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    return enc.decode(idx[0].tolist())


demo = gr.Interface(
    fn=generate_fn,
    inputs=gr.Textbox(lines=3, label="Prompt"),
    outputs=gr.Textbox(lines=10, label="Generated Text"),
    title="GPT-124M (Custom Trained on input.txt)",
    description="Enter a prompt and generate text using your decoder-only transformer trained on the ERA input.txt.",
)

if __name__ == "__main__":
    demo.launch()
