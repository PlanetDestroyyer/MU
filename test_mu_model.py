"""
Test MU Transformer - Interactive Text Generation

Load the trained MU model and generate text interactively!

Usage:
    python test_mu_model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from run_colab import DynamicMUTransformer, Config
import sys

def load_model(model_path='mu_model.pt', device='cuda'):
    """Load trained MU model"""
    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    char_to_idx = checkpoint.get('char_to_idx', None)
    idx_to_char = checkpoint.get('idx_to_char', None)

    # Create model
    model = DynamicMUTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  • Vocab size: {config.vocab_size}")

    return model, config, char_to_idx, idx_to_char


def text_to_tensor(text, char_to_idx, max_len=128):
    """Convert text to tensor"""
    unk_idx = char_to_idx.get('<UNK>', len(char_to_idx) - 1)
    indices = [char_to_idx.get(c, unk_idx) for c in text[:max_len]]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def generate_text(model, prompt, char_to_idx, idx_to_char, max_length=200,
                  temperature=0.8, top_k=40, device='cuda'):
    """
    Generate text from prompt

    Args:
        model: Trained MU model
        prompt: Input text prompt
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top-k tokens (None = sample from all)
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()

    # Convert prompt to tensor
    input_ids = text_to_tensor(prompt, char_to_idx).to(device)

    generated = prompt

    with torch.no_grad():
        for _ in range(max_length - len(prompt)):
            # Get model predictions
            logits = model(input_ids)

            # Get logits for last position
            next_token_logits = logits[0, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Convert to character
            next_char = idx_to_char.get(next_token.item(), '?')
            generated += next_char

            # Stop if we hit max context length
            if input_ids.size(1) >= 128:
                # Keep last 64 tokens
                input_ids = input_ids[:, -64:]

    return generated


def interactive_mode(model, char_to_idx, idx_to_char, device='cuda'):
    """Interactive text generation"""
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE MU TRANSFORMER")
    print("=" * 80)
    print("\nCommands:")
    print("  • Type text to generate continuation")
    print("  • 'temp X' to set temperature (e.g., 'temp 0.5')")
    print("  • 'length X' to set generation length (e.g., 'length 100')")
    print("  • 'quit' or 'exit' to quit")
    print("\nSettings:")

    temperature = 0.8
    max_length = 200
    top_k = 40

    print(f"  • Temperature: {temperature}")
    print(f"  • Max length: {max_length}")
    print(f"  • Top-k: {top_k}")
    print("=" * 80)

    while True:
        try:
            user_input = input("\n📝 Enter prompt: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Handle commands
            if user_input.startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"✓ Temperature set to {temperature}")
                    continue
                except:
                    print("✗ Invalid temperature. Use: temp 0.8")
                    continue

            if user_input.startswith('length '):
                try:
                    max_length = int(user_input.split()[1])
                    print(f"✓ Max length set to {max_length}")
                    continue
                except:
                    print("✗ Invalid length. Use: length 200")
                    continue

            # Generate text
            print(f"\n🤖 Generating (temp={temperature}, len={max_length})...")
            generated = generate_text(
                model, user_input, char_to_idx, idx_to_char,
                max_length=max_length, temperature=temperature,
                top_k=top_k, device=device
            )

            print("\n" + "=" * 80)
            print("GENERATED TEXT:")
            print("=" * 80)
            print(generated)
            print("=" * 80)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            continue


def batch_generation(model, prompts, char_to_idx, idx_to_char, device='cuda'):
    """Generate text for multiple prompts"""
    print("\n" + "=" * 80)
    print("📄 BATCH GENERATION")
    print("=" * 80)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: \"{prompt}\"")
        generated = generate_text(
            model, prompt, char_to_idx, idx_to_char,
            max_length=150, temperature=0.7, top_k=40, device=device
        )
        print(f"Generated:\n{generated}\n")
        print("-" * 80)


def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load model
    try:
        model, config, char_to_idx, idx_to_char = load_model('mu_model.pt', device)
    except FileNotFoundError:
        print("✗ Error: mu_model.pt not found!")
        print("  Please run 'python run_colab.py' first to train the model.")
        sys.exit(1)

    if char_to_idx is None or idx_to_char is None:
        print("✗ Error: Model doesn't have vocabulary mapping!")
        print("  Please retrain the model with the updated run_colab.py")
        sys.exit(1)

    # Choose mode
    print("\n" + "=" * 80)
    print("Choose mode:")
    print("  1. Interactive mode (type prompts)")
    print("  2. Demo with sample prompts")
    print("=" * 80)

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        interactive_mode(model, char_to_idx, idx_to_char, device)
    elif choice == '2':
        # Demo prompts
        prompts = [
            "The quick brown ",
            "Once upon a time",
            "In the beginning",
            "Hello world",
            "The meaning of life"
        ]
        batch_generation(model, prompts, char_to_idx, idx_to_char, device)
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_mode(model, char_to_idx, idx_to_char, device)


if __name__ == '__main__':
    main()
