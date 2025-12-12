"""
Dataset loaders for MU Transformer training and evaluation
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional, Dict, List, Tuple
from transformers import GPT2TokenizerFast
import numpy as np


class WikiTextDataset(Dataset):
    """
    WikiText-2 dataset for language modeling

    Args:
        split: 'train', 'validation', or 'test'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        stride: Stride for creating overlapping sequences
    """

    def __init__(
        self,
        split: str = 'train',
        tokenizer=None,
        max_length: int = 256,
        stride: Optional[int] = None
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length

        # Load dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Initialize tokenizer if not provided
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

        # Tokenize all text
        all_text = ' '.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])

        # Tokenize
        tokenized = tokenizer(
            all_text,
            return_tensors='pt',
            truncation=False,
            add_special_tokens=True
        )

        self.input_ids = tokenized['input_ids'].squeeze(0)

        # Create sequences with stride
        self.sequences = []
        for i in range(0, len(self.input_ids) - max_length, self.stride):
            seq = self.input_ids[i:i + max_length]
            if len(seq) == max_length:
                self.sequences.append(seq)

        print(f"Created {len(self.sequences)} sequences from {split} split")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with 'input_ids' and 'labels'
        """
        seq = self.sequences[idx]

        # For language modeling, labels are shifted input_ids
        return {
            'input_ids': seq[:-1],  # All tokens except last
            'labels': seq[1:]       # All tokens except first (shifted)
        }


class WiCDataset(Dataset):
    """
    Word-in-Context (WiC) dataset for word sense disambiguation

    Args:
        split: 'train', 'validation', or 'test'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        split: str = 'train',
        tokenizer=None,
        max_length: int = 128
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length

        # Load dataset
        dataset = load_dataset("super_glue", "wic", split=split)

        # Initialize tokenizer if not provided
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.data = dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with sentence encodings and label
        """
        item = self.data[idx]

        # Tokenize both sentences
        sent1 = self.tokenizer(
            item['sentence1'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sent2 = self.tokenizer(
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'sent1_input_ids': sent1['input_ids'].squeeze(0),
            'sent1_attention_mask': sent1['attention_mask'].squeeze(0),
            'sent2_input_ids': sent2['input_ids'].squeeze(0),
            'sent2_attention_mask': sent2['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'word': item['word']
        }


class MRPCDataset(Dataset):
    """
    Microsoft Research Paraphrase Corpus for paraphrase detection

    Args:
        split: 'train', 'validation', or 'test'
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        split: str = 'train',
        tokenizer=None,
        max_length: int = 128
    ):
        super().__init__()

        self.split = split
        self.max_length = max_length

        # Load dataset
        dataset = load_dataset("glue", "mrpc", split=split)

        # Initialize tokenizer if not provided
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.data = dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with sentence pair encoding and label
        """
        item = self.data[idx]

        # Tokenize sentence pair
        encoding = self.tokenizer(
            item['sentence1'],
            item['sentence2'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }


def get_tokenizer(vocab_size: int = 30000):
    """
    Get or create tokenizer

    Args:
        vocab_size: Desired vocabulary size

    Returns:
        Tokenizer
    """
    # For simplicity, use GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Note: GPT-2 tokenizer has vocab_size ~50k
    # For exact vocab size control, we would need to train custom tokenizer
    return tokenizer


def get_dataloaders(
    data_config: Dict,
    test_mode: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders

    Args:
        data_config: Data configuration dictionary
        test_mode: If True, use minimal data for testing

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_name = data_config.get('dataset', 'wikitext-2')
    max_length = data_config.get('sequence_length', 256)
    batch_size = data_config.get('batch_size', 32)
    num_workers = data_config.get('num_workers', 4)

    # Get tokenizer
    tokenizer = get_tokenizer(data_config.get('vocab_size', 30000))

    if dataset_name == 'wikitext-2':
        # Language modeling dataset
        train_dataset = WikiTextDataset(
            split='train',
            tokenizer=tokenizer,
            max_length=max_length
        )
        val_dataset = WikiTextDataset(
            split='validation',
            tokenizer=tokenizer,
            max_length=max_length
        )
        test_dataset = WikiTextDataset(
            split='test',
            tokenizer=tokenizer,
            max_length=max_length
        )

    elif dataset_name == 'wic':
        # Word-in-context dataset
        train_dataset = WiCDataset(
            split='train',
            tokenizer=tokenizer,
            max_length=max_length
        )
        val_dataset = WiCDataset(
            split='validation',
            tokenizer=tokenizer,
            max_length=max_length
        )
        # WiC doesn't have public test labels
        test_dataset = val_dataset

    elif dataset_name == 'mrpc':
        # Paraphrase detection dataset
        train_dataset = MRPCDataset(
            split='train',
            tokenizer=tokenizer,
            max_length=max_length
        )
        val_dataset = MRPCDataset(
            split='validation',
            tokenizer=tokenizer,
            max_length=max_length
        )
        test_dataset = MRPCDataset(
            split='test',
            tokenizer=tokenizer,
            max_length=max_length
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # In test mode, use tiny subsets
    if test_mode:
        train_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, range(min(50, len(val_dataset))))
        test_dataset = torch.utils.data.Subset(test_dataset, range(min(50, len(test_dataset))))
        batch_size = 4
        num_workers = 0

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
