"""Debug script to check vocabulary contents"""
import torch
from run_colab import WikiTextDataset, Config

config = Config()

# Load dataset
print("Loading dataset to check vocabulary...")
train_dataset = WikiTextDataset('train', config.max_seq_len, config.vocab_size)

print(f"\n{'='*80}")
print(f"VOCABULARY ANALYSIS")
print(f"{'='*80}")
print(f"Total vocab size: {len(train_dataset.char_to_idx)}")

# Get all characters sorted by index
chars_by_idx = [(idx, char) for char, idx in train_dataset.char_to_idx.items()]
chars_by_idx.sort(key=lambda x: x[0])

print(f"\nFirst 100 characters in vocabulary:")
for i, (idx, char) in enumerate(chars_by_idx[:100]):
    if char in ['<PAD>', '<UNK>']:
        print(f"{idx:4d}: {char}")
    else:
        print(f"{idx:4d}: '{char}' (Unicode: U+{ord(char):04X})")

print(f"\n\nLast 50 characters in vocabulary:")
for idx, char in chars_by_idx[-50:]:
    if char in ['<PAD>', '<UNK>']:
        print(f"{idx:4d}: {char}")
    else:
        print(f"{idx:4d}: '{char}' (Unicode: U+{ord(char):04X})")

# Count character types
english_letters = sum(1 for _, ch in chars_by_idx if ch.isalpha() and ord(ch) < 128)
digits = sum(1 for _, ch in chars_by_idx if ch.isdigit())
punctuation = sum(1 for _, ch in chars_by_idx if not ch.isalnum() and ord(ch) < 128 and ch not in ['<PAD>', '<UNK>'])
unicode_chars = sum(1 for _, ch in chars_by_idx if ord(ch) >= 128 and ch not in ['<PAD>', '<UNK>'])

print(f"\n{'='*80}")
print("CHARACTER TYPE BREAKDOWN:")
print(f"{'='*80}")
print(f"English letters (a-z, A-Z): {english_letters}")
print(f"Digits (0-9): {digits}")
print(f"ASCII punctuation: {punctuation}")
print(f"Non-ASCII Unicode characters: {unicode_chars}")
print(f"Special tokens (<PAD>, <UNK>): 2")
print(f"Total: {len(train_dataset.char_to_idx)}")

print(f"\n{'='*80}")
print("DIAGNOSIS:")
print(f"{'='*80}")
if unicode_chars > 100:
    print(f"⚠️  WARNING: {unicode_chars} Unicode characters detected!")
    print("   This explains why generation produces multilingual gibberish.")
    print("   WikiText-2 contains citations, names, and special characters.")
    print("\n   SOLUTION: Filter vocabulary to English-only (a-z, A-Z, 0-9, punctuation)")
else:
    print("✓ Vocabulary looks clean (mostly ASCII)")
