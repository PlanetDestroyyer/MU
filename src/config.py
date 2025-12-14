"""Configuration for MU-SOTA Transformer"""

import torch


class MUSOTAConfig:
    """Configuration for MU-SOTA Transformer"""

    # Matrix structure (8×8 = 64 dims, 16 semantic blocks)
    matrix_size = 8
    num_semantic_blocks = 16  # Each block is 2×2
    block_size = 2  # 2×2 blocks

    # Architecture (SOTA-level)
    n_layers = 24  # Deep like GPT-2 Medium
    n_heads = 8
    dropout = 0.1

    # Vocabulary
    vocab_size = 50000  # Like GPT-2
    max_seq_len = 512

    # Training
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 0.01
    warmup_steps = 1000
    max_grad_norm = 1.0

    # Mixed precision
    use_mixed_precision = True

    # Generation
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.2

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
