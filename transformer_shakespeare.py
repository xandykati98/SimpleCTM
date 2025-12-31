import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import wandb
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional

# --- Data Loading ---

def get_shakespeare_data(data_path: str) -> str:
    """Download and return the Tiny Shakespeare dataset."""
    file_path: str = os.path.join(data_path, "input.txt")
    if not os.path.exists(file_path):
        os.makedirs(data_path, exist_ok=True)
        url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare dataset from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)
    
    with open(file_path, "r", encoding="utf-8") as f:
        text: str = f.read()
    
    return text

class ShakespeareDataset(Dataset):
    """Dataset for character-level language modeling."""
    def __init__(self, text: str, block_size: int):
        chars: List[str] = sorted(list(set(text)))
        self.vocab_size: int = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        self.block_size: int = block_size
        self.data: torch.Tensor = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk: torch.Tensor = self.data[idx : idx + self.block_size + 1]
        x: torch.Tensor = chunk[:-1]
        y: torch.Tensor = chunk[1:]
        return x, y

# --- Transformer Components ---

class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.query: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.value: nn.Linear = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k: torch.Tensor = self.key(x)   # (B, T, head_size)
        q: torch.Tensor = self.query(x) # (B, T, head_size)
        
        # Compute attention scores ("affinities")
        wei: torch.Tensor = q @ k.transpose(-2, -1) * (C**-0.5) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation
        v: torch.Tensor = self.value(x) # (B, T, head_size)
        out: torch.Tensor = wei @ v    # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""
    def __init__(self, n_heads: int, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.heads: nn.ModuleList = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_heads)])
        self.proj: nn.Linear = nn.Linear(n_embd, n_embd)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Simple linear layer followed by a non-linearity."""
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size: int = n_embd // n_heads
        self.sa: MultiHeadAttention = MultiHeadAttention(n_heads, n_embd, head_size, block_size, dropout)
        self.ffwd: FeedForward = FeedForward(n_embd, dropout)
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ShakespeareTransformer(nn.Module):
    """Full GPT-style Transformer model."""
    def __init__(self, vocab_size: int, n_embd: int, n_heads: int, n_layers: int, block_size: int, dropout: float):
        super().__init__()
        self.token_embedding_table: nn.Embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table: nn.Embedding = nn.Embedding(block_size, n_embd)
        self.blocks: nn.Sequential = nn.Sequential(*[Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)])
        self.ln_f: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.lm_head: nn.Linear = nn.Linear(n_embd, vocab_size)
        self.block_size: int = block_size

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        tok_emb: torch.Tensor = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb: torch.Tensor = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x: torch.Tensor = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits: torch.Tensor = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped: torch.Tensor = logits.view(B*T, C)
            targets_reshaped: torch.Tensor = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond: torch.Tensor = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs: torch.Tensor = F.softmax(logits, dim=-1) # (B, C)
            idx_next: torch.Tensor = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- Main Training Function ---

def main(data_path: str = "./data", checkpoint_dir: str = "."):
    # Hyperparameters
    batch_size: int = 64
    block_size: int = 256
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_iters: int = 200
    n_embd: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.2
    
    print(f"Using device: {device}")
    
    text: str = get_shakespeare_data(data_path)
    dataset: ShakespeareDataset = ShakespeareDataset(text, block_size)
    vocab_size: int = dataset.vocab_size
    
    # Split data
    n: int = int(0.9 * len(text))
    train_data: str = text[:n]
    val_data: str = text[n:]
    
    train_dataset: ShakespeareDataset = ShakespeareDataset(train_data, block_size)
    val_dataset: ShakespeareDataset = ShakespeareDataset(val_data, block_size)
    
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model: ShakespeareTransformer = ShakespeareTransformer(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    wandb.init(
        project="shakespeare-transformer",
        config={
            "batch_size": batch_size,
            "block_size": block_size,
            "learning_rate": learning_rate,
            "n_embd": n_embd,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dropout": dropout,
            "vocab_size": vocab_size,
        }
    )
    
    print("Starting training...")
    model.train()
    
    # Iterative training loop
    data_iter = iter(train_loader)
    for i in range(max_iters):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)
            
        xb, yb = xb.to(device), yb.to(device)
        
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % eval_interval == 0 or i == max_iters - 1:
            model.eval()
            with torch.no_grad():
                val_losses: List[float] = []
                val_iter = iter(val_loader)
                for _ in range(min(eval_iters, len(val_loader))):
                    try:
                        xv, yv = next(val_iter)
                        xv, yv = xv.to(device), yv.to(device)
                        _, v_loss = model(xv, yv)
                        val_losses.append(v_loss.item())
                    except StopIteration:
                        break
                avg_val_loss: float = sum(val_losses) / len(val_losses) if val_losses else 0.0
            
            print(f"Step {i}: train loss {loss.item():.4f}, val loss {avg_val_loss:.4f}")
            wandb.log({"train_loss": loss.item(), "val_loss": avg_val_loss, "step": i})
            
            # Generate sample text
            context: torch.Tensor = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated: List[int] = model.generate(context, max_new_tokens=100)[0].tolist()
            sample_text: str = "".join([dataset.itos[idx] for idx in generated])
            print(f"Sample generation:\n{sample_text}\n" + "-"*30)
            
            model.train()

    # Save model
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path: str = os.path.join(checkpoint_dir, "shakespeare_transformer.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()

