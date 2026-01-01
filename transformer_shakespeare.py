import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import wandb
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class ShakespeareDataset(Dataset):
    """Shakespeare dataset using custom BPE tokenization for efficiency."""
    
    def __init__(self, data_path: str, seq_len: int, split: str = 'train', vocab_size: int = 1024):
        self.seq_len = seq_len
        self.tokenizer_path = os.path.join(data_path, f'tokenizer_shakespeare_bytelevel_{vocab_size}.json')
        
        # Download or load Shakespeare text
        shakespeare_path = os.path.join(data_path, 'shakespeare.txt')
        if not os.path.exists(shakespeare_path):
            os.makedirs(data_path, exist_ok=True)
            import urllib.request
            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            urllib.request.urlretrieve(url, shakespeare_path)
        
        # Train or load tokenizer
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        else:
            print(f"Training new ByteLevel BPE tokenizer with vocab size {vocab_size}...")
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            
            trainer = BpeTrainer(
                special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"], 
                vocab_size=vocab_size,
                initial_alphabet=ByteLevel.alphabet()
            )
            self.tokenizer.train([shakespeare_path], trainer)
            self.tokenizer.save(self.tokenizer_path)
            print("Tokenizer trained and saved.")

        self.vocab_size = self.tokenizer.get_vocab_size()
        
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Encode text
        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded.ids, dtype=torch.long)
        
        # Split data
        n = len(self.data)
        train_data = self.data[:int(n * 0.9)]
        val_data = self.data[int(n * 0.9):]
        
        self.data = train_data if split == 'train' else val_data
    
    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y
    
    def decode(self, indices: torch.Tensor) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return self.tokenizer.decode(indices)
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)


class TransformerBlock(nn.Module):
    """Standard Transformer Block."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """GPT-style Transformer for language modeling."""
    
    def __init__(
        self, 
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.token_embed.weight = self.head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        device = x.device
        
        # Embeddings
        x = self.token_embed(x) + self.pos_embed[:, :S, :]
        x = self.dropout(x)
        
        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=device), diagonal=1).bool()
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            context = generated[:, -self.max_seq_len:]
            logits = self(context)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


def train_model(
    model: TransformerLM,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
    device: torch.device,
    dataset: ShakespeareDataset,
    val_loader: DataLoader | None = None,
    use_wandb: bool = False,
):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            
            # Batch-level statistics for wandb
            if use_wandb:
                batch_perplexity = math.exp(loss.item())
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/perplexity": batch_perplexity,
                    "epoch": epoch,
                })
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_total_tokens = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    v_logits = model(vx)
                    v_loss = F.cross_entropy(v_logits.reshape(-1, v_logits.size(-1)), vy.reshape(-1))
                    val_total_loss += v_loss.item() * vx.numel()
                    val_total_tokens += vx.numel()
            val_loss = val_total_loss / val_total_tokens
            print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}')
            model.train()

        if use_wandb:
            wandb.log({"train/loss": avg_loss, "val/loss": val_loss, "epoch": epoch})
            
        # Sample generation
        model.eval()
        prompt = "Alexander then said "
        prompt_tokens = dataset.encode(prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8)
        print(f"\n--- Epoch {epoch+1} Sample ---")
        print(dataset.decode(generated[0]))
        print("----------------------------\n")
        model.train()


def main(data_path: str = './data', checkpoint_dir: str = '.'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    seq_len = 64
    batch_size = 32
    target_vocab_size = 1024
    
    train_dataset = ShakespeareDataset(data_path=data_path, seq_len=seq_len, split='train', vocab_size=target_vocab_size)
    val_dataset = ShakespeareDataset(data_path=data_path, seq_len=seq_len, split='val', vocab_size=target_vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Matching CTM model size roughly (d_model=128)
    model = TransformerLM(
        vocab_size=train_dataset.vocab_size,
        max_seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    wandb.init(project="shakespeare", name="transformer_baseline")
    
    train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=2,
        device=device,
        dataset=train_dataset,
        val_loader=val_loader,
        use_wandb=True
    )
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'transformer_shakespeare.pth'))
    wandb.finish()


if __name__ == '__main__':
    main()

