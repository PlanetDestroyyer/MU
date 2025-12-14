"""Configuration for MU-SOTA Transformer"""

import torch


class MUSOTAConfig:
    """Configuration for MU-SOTA Transformer"""

    # Matrix structure (8×8 = 64 dims, 16 semantic blocks)
    matrix_size = 8
    num_semantic_blocks = 16  # Each block is 2×2
    block_size = 2  # 2×2 blocks

    # Architecture (SOTA-level)
    n_layers = 12  # Scaled up from 6 for better performance
    n_heads = 8
    dropout = 0.1

    # Vocabulary
    vocab_size = 50000  # Like GPT-2
    max_seq_len = 512  # Optimized for memory efficiency

    # Training
    batch_size = 32 # Adjusted for 12-layer model memory requirements
    num_epochs = 10  # Increased for better convergence
    learning_rate = 3e-4  # Slightly higher for faster convergence
    weight_decay = 0.01
    warmup_steps = 500  # Reduced warmup
    max_grad_norm = 1.0
    gradient_accumulation_steps = 2  # Simulate batch_size=32

    # Mixed precision
    use_mixed_precision = True

    # Generation
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.2

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
