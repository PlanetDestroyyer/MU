"""
Data augmentation for stability testing
"""
import torch
import numpy as np
from typing import List, Optional
import random


class TextAugmenter:
    """
    Text augmentation for testing embedding stability

    Args:
        dropout_prob: Probability of dropping a token
        swap_prob: Probability of swapping adjacent tokens
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        swap_prob: float = 0.1,
        seed: Optional[int] = None
    ):
        self.dropout_prob = dropout_prob
        self.swap_prob = swap_prob

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def random_token_dropout(
        self,
        input_ids: torch.Tensor,
        mask_token_id: int = 50256  # GPT-2 <|endoftext|> token
    ) -> torch.Tensor:
        """
        Randomly drop tokens from sequence

        Args:
            input_ids: [T] or [B, T] tensor of token IDs
            mask_token_id: Token to use as replacement

        Returns:
            Augmented input_ids
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T = input_ids.shape
        augmented = input_ids.clone()

        # Create dropout mask
        dropout_mask = torch.rand(B, T) < self.dropout_prob

        # Replace dropped tokens
        augmented[dropout_mask] = mask_token_id

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def random_token_swap(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Randomly swap adjacent tokens

        Args:
            input_ids: [T] or [B, T] tensor of token IDs

        Returns:
            Augmented input_ids
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T = input_ids.shape
        augmented = input_ids.clone()

        for b in range(B):
            for t in range(T - 1):
                if random.random() < self.swap_prob:
                    # Swap t and t+1
                    temp = augmented[b, t].item()
                    augmented[b, t] = augmented[b, t + 1]
                    augmented[b, t + 1] = temp

        if squeeze:
            augmented = augmented.squeeze(0)

        return augmented

    def augment(
        self,
        input_ids: torch.Tensor,
        methods: List[str] = ['dropout', 'swap']
    ) -> torch.Tensor:
        """
        Apply multiple augmentation methods

        Args:
            input_ids: [T] or [B, T] tensor of token IDs
            methods: List of augmentation methods to apply

        Returns:
            Augmented input_ids
        """
        augmented = input_ids.clone()

        for method in methods:
            if method == 'dropout':
                augmented = self.random_token_dropout(augmented)
            elif method == 'swap':
                augmented = self.random_token_swap(augmented)
            else:
                raise ValueError(f"Unknown augmentation method: {method}")

        return augmented


class SynonymReplacer:
    """
    Replace words with synonyms using WordNet

    Note: This is a placeholder implementation.
    Full implementation would require NLTK and WordNet.
    """

    def __init__(self):
        self.synonyms = {}  # Word -> list of synonyms

    def replace_synonyms(
        self,
        text: str,
        replacement_prob: float = 0.1
    ) -> str:
        """
        Replace words with synonyms

        Args:
            text: Input text
            replacement_prob: Probability of replacing each word

        Returns:
            Text with some words replaced by synonyms
        """
        # Placeholder implementation
        # Full version would use NLTK WordNet
        words = text.split()
        for i, word in enumerate(words):
            if random.random() < replacement_prob:
                # In real implementation, look up synonyms
                # For now, just keep original
                pass

        return ' '.join(words)


def create_augmented_pairs(
    input_ids: torch.Tensor,
    augmenter: TextAugmenter,
    num_augmentations: int = 1
) -> List[torch.Tensor]:
    """
    Create multiple augmented versions of input

    Args:
        input_ids: Original input IDs [B, T]
        augmenter: TextAugmenter instance
        num_augmentations: Number of augmented versions to create

    Returns:
        List of augmented input_ids tensors
    """
    augmented_list = [input_ids]  # Include original

    for _ in range(num_augmentations):
        augmented = augmenter.augment(input_ids)
        augmented_list.append(augmented)

    return augmented_list
