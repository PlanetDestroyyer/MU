"""
Data preprocessing utilities
"""
import torch
import numpy as np
from typing import List, Dict, Optional


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create attention mask from input IDs

    Args:
        input_ids: [B, T] tensor of token IDs
        pad_token_id: ID of padding token

    Returns:
        Attention mask [B, T] where 1 = attend, 0 = ignore
    """
    return (input_ids != pad_token_id).long()


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive models

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Causal mask [seq_len, seq_len] where True = attend, False = mask
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_length: Optional[int] = None
) -> torch.Tensor:
    """
    Pad sequences to same length

    Args:
        sequences: List of 1D tensors
        padding_value: Value to use for padding
        max_length: Maximum length (if None, use longest sequence)

    Returns:
        Padded tensor [len(sequences), max_length]
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        if len(seq) < max_length:
            padding = torch.full((max_length - len(seq),), padding_value, dtype=seq.dtype)
            padded.append(torch.cat([seq, padding]))
        else:
            padded.append(seq[:max_length])

    return torch.stack(padded)


def truncate_sequence(
    sequence: torch.Tensor,
    max_length: int,
    truncation_strategy: str = 'longest_first'
) -> torch.Tensor:
    """
    Truncate sequence to maximum length

    Args:
        sequence: Input sequence
        max_length: Maximum length
        truncation_strategy: 'longest_first', 'only_first', or 'only_second'

    Returns:
        Truncated sequence
    """
    if len(sequence) <= max_length:
        return sequence

    if truncation_strategy == 'longest_first':
        return sequence[:max_length]
    else:
        return sequence[:max_length]


class Collator:
    """
    Custom collator for batching sequences

    Args:
        pad_token_id: Padding token ID
        max_length: Maximum sequence length
    """

    def __init__(self, pad_token_id: int = 0, max_length: int = 256):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched dictionary
        """
        # Handle different dataset types
        if 'input_ids' in batch[0] and 'labels' in batch[0]:
            # Language modeling dataset
            input_ids = [item['input_ids'] for item in batch]
            labels = [item['labels'] for item in batch]

            input_ids = pad_sequence(input_ids, self.pad_token_id, self.max_length)
            labels = pad_sequence(labels, -100, self.max_length)  # -100 is ignore index

            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': create_attention_mask(input_ids, self.pad_token_id)
            }

        else:
            # Other datasets - just stack
            return {
                key: torch.stack([item[key] for item in batch])
                for key in batch[0].keys()
                if isinstance(batch[0][key], torch.Tensor)
            }
