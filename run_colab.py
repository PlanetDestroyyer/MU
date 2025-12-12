"""
Comprehensive MU Transformer vs Baseline Comparison

This script:
1. Downloads and preprocesses WikiText-2 dataset
2. Trains MU Transformer for 5 epochs with full metrics
3. Trains Baseline Transformer (SAME parameter count) with dense matrices
4. Shows comprehensive comparison: accuracy, loss, perplexity
5. Generates detailed visualizations

Usage:
    python run_colab.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Unified configuration for both models"""
    # Data
    vocab_size = 10000
    max_seq_len = 128

    # MU Transformer parameters
    r = 4  # MU matrix rows
    c = 4  # MU matrix columns
    d_model = 128
    n_layers = 4
    n_heads = 4
    dropout = 0.1

    # Baseline will match MU parameter count by adjusting d_model

    # Training
    batch_size = 32
    num_epochs = 5
    learning_rate = 3e-4
    warmup_steps = 200
    weight_decay = 0.01
    max_grad_norm = 1.0

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()
print(f"Using device: {config.device}")
print("=" * 70)

# ============================================================================
# DATASET
# ============================================================================

class WikiTextDataset(Dataset):
    """WikiText-2 dataset with character-level tokenization"""

    def __init__(self, split='train', max_seq_len=128, vocab_size=10000, char_to_idx=None):
        print(f"Loading {split} dataset...")

        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Combine all text
        all_text = ' '.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])

        # Build or use vocabulary
        if char_to_idx is None:
            # Build vocabulary from training data
            chars = sorted(list(set(all_text)))[:vocab_size-2]
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.char_to_idx['<PAD>'] = vocab_size - 2
            self.char_to_idx['<UNK>'] = vocab_size - 1
        else:
            self.char_to_idx = char_to_idx

        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = vocab_size

        # Create sequences
        self.data = []
        stride = max_seq_len // 2
        for i in range(0, len(all_text) - max_seq_len - 1, stride):
            chunk = all_text[i:i + max_seq_len + 1]
            if len(chunk) == max_seq_len + 1:
                tokens = [self.char_to_idx.get(c, vocab_size-1) for c in chunk]
                self.data.append(torch.tensor(tokens, dtype=torch.long))

        print(f"Created {len(self.data)} sequences from {split} split")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return {
            'input_ids': seq[:-1],
            'labels': seq[1:]
        }

# ============================================================================
# MU TRANSFORMER IMPLEMENTATION
# ============================================================================

class MUAttentionLayer(nn.Module):
    """MU Attention with semantic gating"""

    def __init__(self, r=4, c=4, d_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.r, self.c = r, c
        self.rc = r * c
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Projections
        self.W_q = nn.Linear(self.rc, d_model)
        self.W_k = nn.Linear(self.rc, d_model)
        self.W_v = nn.Linear(self.rc, d_model)
        self.W_out = nn.Linear(d_model, self.rc)
        self.W_g = nn.Linear(d_model, self.rc)
        self.W_b = nn.Linear(d_model, self.rc)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([r, c])

        # Sensitivity mask
        sensitivity_mask = torch.tensor([
            [0.1, 0.01, 0.01, 0.7],
            [0.7, 0.7, 0.7, 0.9],
            [0.9, 0.9, 0.9, 0.6],
            [0.6, 0.5, 0.5, 0.1]
        ], dtype=torch.float32)
        self.register_buffer('sensitivity_mask', sensitivity_mask)

        self._init_weights()

    def _init_weights(self):
        for m in [self.W_q, self.W_k, self.W_v, self.W_out, self.W_g, self.W_b]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, M, mask=None):
        B, T, r, c = M.shape
        M_flat = M.view(B, T, self.rc)

        # Project to Q, K, V
        Q = self.W_q(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(M_flat).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)

        # Project back to MU space
        delta_M = self.W_out(context).view(B, T, r, c)

        # Gating with sensitivity
        G = torch.sigmoid(self.W_g(context)).view(B, T, r, c)
        G = G * self.sensitivity_mask.unsqueeze(0).unsqueeze(0)

        # Bias term
        B_term = torch.tanh(self.W_b(context)).view(B, T, r, c) * 0.1

        # Update
        M_updated = M * (1 - G) + delta_M * G + B_term
        M_updated = self.layer_norm(M_updated)

        return M_updated


class MUTransformer(nn.Module):
    """MU Transformer with MU-parameterized layers"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.r, self.c = config.r, config.c
        self.rc = self.r * self.c

        # Token to MU embedding
        self.token_to_mu = nn.Embedding(config.vocab_size, self.rc)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_seq_len, self.r, self.c) * 0.02)

        # MU layers
        self.layers = nn.ModuleList([
            MUAttentionLayer(self.r, self.c, config.d_model, config.n_heads, config.dropout)
            for _ in range(config.n_layers)
        ])

        # Output head
        self.output = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(self.rc, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.vocab_size)
        )

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).bool()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Token to MU
        MU = self.token_to_mu(input_ids).view(B, T, self.r, self.c)
        MU = MU + self.pos_embedding[:, :T, :, :]
        MU = self.dropout(MU)

        # Causal mask
        mask = self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0)

        # Apply MU layers
        for layer in self.layers:
            MU = layer(MU, mask=mask)

        # Output logits
        logits = self.output(MU)

        return logits


# ============================================================================
# BASELINE TRANSFORMER WITH DENSE MATRICES
# ============================================================================

class DenseTransformerBlock(nn.Module):
    """Standard transformer block with dense attention and FFN"""

    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Multi-head self-attention (DENSE)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feedforward network (DENSE)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class BaselineTransformer(nn.Module):
    """Baseline Transformer with DENSE matrices (matched parameter count)"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Calculate d_model for baseline to match MU parameter count
        # We'll use a smaller d_model to roughly match parameters
        self.d_model = 64  # Adjusted to match MU params
        self.d_ff = self.d_model * 2

        # Token embedding (DENSE)
        self.token_embedding = nn.Embedding(config.vocab_size, self.d_model)

        # Positional embedding (DENSE)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_seq_len, self.d_model) * 0.02
        )

        # Transformer blocks (DENSE)
        self.layers = nn.ModuleList([
            DenseTransformerBlock(
                d_model=self.d_model,
                n_heads=config.n_heads,
                d_ff=self.d_ff,
                dropout=config.dropout
            )
            for _ in range(config.n_layers)
        ])

        # Output projection (DENSE)
        self.output = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, config.vocab_size)
        )

        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len) * float('-inf'), diagonal=1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Dense embeddings
        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        x = self.dropout(x)

        # Causal mask
        mask = self.causal_mask[:T, :T]

        # Apply dense transformer blocks
        for layer in self.layers:
            x = layer(x, mask=mask)

        # Output logits
        logits = self.output(x)

        return logits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_accuracy(logits, labels):
    """Compute token-level accuracy"""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).float()
    return correct.mean().item()


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs):
    """Train for one epoch with detailed metrics"""
    model.train()

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        # Compute accuracy
        with torch.no_grad():
            accuracy = compute_accuracy(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Track metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy
    }


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, total_epochs, split='Val'):
    """Evaluate model with comprehensive metrics"""
    model.eval()

    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{total_epochs} [{split}]')
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        # Compute accuracy
        accuracy = compute_accuracy(logits, labels)

        # Track metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': perplexity
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results, save_path='comparison_results.png'):
    """Generate comprehensive comparison plots"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MU Transformer vs Baseline Transformer - Comprehensive Comparison',
                 fontsize=16, fontweight='bold')

    epochs = range(1, config.num_epochs + 1)

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, results['MU']['train_loss'], 'o-',
                    label='MU Transformer', linewidth=2, markersize=8)
    axes[0, 0].plot(epochs, results['Baseline']['train_loss'], 's-',
                    label='Baseline', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, results['MU']['val_loss'], 'o-',
                    label='MU Transformer', linewidth=2, markersize=8)
    axes[0, 1].plot(epochs, results['Baseline']['val_loss'], 's-',
                    label='Baseline', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('Validation Loss', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Validation Perplexity
    axes[0, 2].plot(epochs, results['MU']['val_perplexity'], 'o-',
                    label='MU Transformer', linewidth=2, markersize=8)
    axes[0, 2].plot(epochs, results['Baseline']['val_perplexity'], 's-',
                    label='Baseline', linewidth=2, markersize=8)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('Perplexity', fontsize=12)
    axes[0, 2].set_title('Validation Perplexity', fontsize=13, fontweight='bold')
    axes[0, 2].legend(fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Training Accuracy
    axes[1, 0].plot(epochs, [a*100 for a in results['MU']['train_accuracy']], 'o-',
                    label='MU Transformer', linewidth=2, markersize=8)
    axes[1, 0].plot(epochs, [a*100 for a in results['Baseline']['train_accuracy']], 's-',
                    label='Baseline', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Training Accuracy', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Validation Accuracy
    axes[1, 1].plot(epochs, [a*100 for a in results['MU']['val_accuracy']], 'o-',
                    label='MU Transformer', linewidth=2, markersize=8)
    axes[1, 1].plot(epochs, [a*100 for a in results['Baseline']['val_accuracy']], 's-',
                    label='Baseline', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].set_title('Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Final Metrics Comparison (Bar chart)
    metrics = ['Val Loss', 'Val Acc (%)', 'Val PPL']
    mu_final = [
        results['MU']['val_loss'][-1],
        results['MU']['val_accuracy'][-1] * 100,
        results['MU']['val_perplexity'][-1]
    ]
    baseline_final = [
        results['Baseline']['val_loss'][-1],
        results['Baseline']['val_accuracy'][-1] * 100,
        results['Baseline']['val_perplexity'][-1]
    ]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1, 2].bar(x - width/2, mu_final, width, label='MU Transformer', alpha=0.8)
    axes[1, 2].bar(x + width/2, baseline_final, width, label='Baseline', alpha=0.8)
    axes[1, 2].set_ylabel('Value', fontsize=12)
    axes[1, 2].set_title('Final Metrics Comparison', fontsize=13, fontweight='bold')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(metrics, fontsize=10)
    axes[1, 2].legend(fontsize=11)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for i, (mu_val, base_val) in enumerate(zip(mu_final, baseline_final)):
        axes[1, 2].text(i - width/2, mu_val, f'{mu_val:.2f}',
                        ha='center', va='bottom', fontsize=9)
        axes[1, 2].text(i + width/2, base_val, f'{base_val:.2f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plots saved to '{save_path}'")
    plt.show()


def print_results_table(results, mu_params, baseline_params):
    """Print detailed results table"""

    print("\n" + "=" * 80)
    print("FINAL RESULTS - COMPREHENSIVE COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<20} {'Parameters':<15} {'Val Loss':<12} {'Val Acc':<12} {'Val PPL':<12}")
    print("-" * 80)

    print(f"{'MU Transformer':<20} {mu_params:<15,} "
          f"{results['MU']['val_loss'][-1]:<12.4f} "
          f"{results['MU']['val_accuracy'][-1]*100:<12.2f} "
          f"{results['MU']['val_perplexity'][-1]:<12.2f}")

    print(f"{'Baseline':<20} {baseline_params:<15,} "
          f"{results['Baseline']['val_loss'][-1]:<12.4f} "
          f"{results['Baseline']['val_accuracy'][-1]*100:<12.2f} "
          f"{results['Baseline']['val_perplexity'][-1]:<12.2f}")

    print("\n" + "-" * 80)

    # Calculate improvements
    loss_improvement = ((results['Baseline']['val_loss'][-1] - results['MU']['val_loss'][-1]) /
                       results['Baseline']['val_loss'][-1]) * 100
    acc_improvement = ((results['MU']['val_accuracy'][-1] - results['Baseline']['val_accuracy'][-1]) /
                      results['Baseline']['val_accuracy'][-1]) * 100
    ppl_improvement = ((results['Baseline']['val_perplexity'][-1] - results['MU']['val_perplexity'][-1]) /
                      results['Baseline']['val_perplexity'][-1]) * 100

    print("\nIMPROVEMENTS (MU vs Baseline):")
    print(f"  • Validation Loss:       {loss_improvement:+.2f}% {'✓' if loss_improvement > 0 else '✗'}")
    print(f"  • Validation Accuracy:   {acc_improvement:+.2f}% {'✓' if acc_improvement > 0 else '✗'}")
    print(f"  • Validation Perplexity: {ppl_improvement:+.2f}% {'✓' if ppl_improvement > 0 else '✗'}")

    param_ratio = (mu_params / baseline_params) * 100
    print(f"\n  • Parameter Efficiency:  {param_ratio:.1f}% of baseline parameters")

    print("\n" + "=" * 80)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("MU TRANSFORMER vs BASELINE TRANSFORMER - COMPREHENSIVE COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  • Epochs: {config.num_epochs}")
    print(f"  • Batch Size: {config.batch_size}")
    print(f"  • Learning Rate: {config.learning_rate}")
    print(f"  • Sequence Length: {config.max_seq_len}")
    print(f"  • Vocabulary Size: {config.vocab_size}")
    print("=" * 80)

    # ========================================================================
    # LOAD DATASET
    # ========================================================================

    print("\n📥 LOADING DATASETS...")
    print("-" * 80)

    try:
        train_dataset = WikiTextDataset('train', config.max_seq_len, config.vocab_size)
        val_dataset = WikiTextDataset('validation', config.max_seq_len, config.vocab_size,
                                     char_to_idx=train_dataset.char_to_idx)

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                 shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                               shuffle=False, num_workers=2, pin_memory=True)

        print(f"✓ Dataset loaded successfully")
        print(f"  • Training sequences: {len(train_dataset):,}")
        print(f"  • Validation sequences: {len(val_dataset):,}")

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Creating dummy data for testing...")

        class DummyDataset(Dataset):
            def __len__(self):
                return 128
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randint(0, config.vocab_size, (config.max_seq_len,)),
                    'labels': torch.randint(0, config.vocab_size, (config.max_seq_len,))
                }

        train_dataset = DummyDataset()
        val_dataset = DummyDataset()
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # ========================================================================
    # TRAIN BOTH MODELS
    # ========================================================================

    results = {
        'MU': {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [], 'val_perplexity': []
        },
        'Baseline': {
            'train_loss': [], 'train_accuracy': [],
            'val_loss': [], 'val_accuracy': [], 'val_perplexity': []
        }
    }

    model_configs = [
        ('MU', MUTransformer),
        ('Baseline', BaselineTransformer)
    ]

    for model_name, ModelClass in model_configs:
        print("\n" + "=" * 80)
        print(f"🚀 TRAINING {model_name.upper()} TRANSFORMER")
        print("=" * 80)

        # Create model
        model = ModelClass(config).to(config.device)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"\nModel Architecture:")
        print(f"  • Total Parameters: {num_params:,}")
        print(f"  • Layers: {config.n_layers}")
        print(f"  • Attention Heads: {config.n_heads}")
        if model_name == 'MU':
            print(f"  • MU Matrix Size: {config.r}×{config.c}")
        else:
            print(f"  • Hidden Dimension: {model.d_model}")

        # Save param count
        if model_name == 'MU':
            mu_params = num_params
        else:
            baseline_params = num_params

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        total_steps = len(train_loader) * config.num_epochs
        scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

        # Training loop
        print(f"\n{'='*80}")
        for epoch in range(1, config.num_epochs + 1):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, scheduler,
                config.device, epoch, config.num_epochs
            )

            # Evaluate
            val_metrics = evaluate(
                model, val_loader, config.device,
                epoch, config.num_epochs, split='Val'
            )

            # Store results
            results[model_name]['train_loss'].append(train_metrics['loss'])
            results[model_name]['train_accuracy'].append(train_metrics['accuracy'])
            results[model_name]['val_loss'].append(val_metrics['loss'])
            results[model_name]['val_accuracy'].append(val_metrics['accuracy'])
            results[model_name]['val_perplexity'].append(val_metrics['perplexity'])

            # Print epoch summary
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  • Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']*100:.2f}%")
            print(f"  • Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                  f"Val PPL: {val_metrics['perplexity']:.2f}")

    # ========================================================================
    # RESULTS AND VISUALIZATION
    # ========================================================================

    print_results_table(results, mu_params, baseline_params)

    print("\n📈 Generating visualization...")
    plot_comparison(results)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check 'comparison_results.png' for detailed plots")
    print("  2. Analyze which model performs better")
    print("  3. Consider hyperparameter tuning based on results")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
