"""WikiText-2 dataset with BPE tokenization"""

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class WikiTextBPEDataset(Dataset):
    """WikiText-2 with BPE tokenization (50K vocab)"""

    def __init__(self, split: str = 'train', max_seq_len: int = 512,
                 tokenizer: Optional[Tokenizer] = None, vocab_size: int = 50000):
        logger.info(f"Loading {split} dataset...")

        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            all_text = ' '.join([item['text'] for item in dataset if len(item['text'].strip()) > 0])

            if tokenizer is None:
                logger.info(f"Training BPE tokenizer with vocab_size={vocab_size}...")
                self.tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
                self.tokenizer.pre_tokenizer = Whitespace()

                trainer = BpeTrainer(
                    vocab_size=vocab_size,
                    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
                    show_progress=False
                )

                self.tokenizer.train_from_iterator([all_text], trainer=trainer)
                self.vocab_size = self.tokenizer.get_vocab_size()
                logger.info(f"Tokenizer trained. Vocab size: {self.vocab_size}")
            else:
                self.tokenizer = tokenizer
                self.vocab_size = self.tokenizer.get_vocab_size()

            # Tokenize text
            encoding = self.tokenizer.encode(all_text)
            all_tokens = encoding.ids

            # Create sequences
            self.data = []
            stride = max_seq_len // 2
            for i in range(0, len(all_tokens) - max_seq_len - 1, stride):
                chunk = all_tokens[i:i + max_seq_len + 1]
                if len(chunk) == max_seq_len + 1:
                    self.data.append(torch.tensor(chunk, dtype=torch.long))

            logger.info(f"Created {len(self.data)} sequences from {split} split")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self.data[idx]
        return {'input_ids': seq[:-1], 'labels': seq[1:]}

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids)
