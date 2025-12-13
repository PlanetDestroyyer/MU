# Testing the MU Transformer

## 🎮 Interactive Text Generation

The trained MU model is saved and ready to test like a normal LLM!

---

## 📋 Prerequisites

The model should already be trained and saved as `mu_model.pt` from running:
```bash
python run_colab.py
```

After training completes, you'll see:
```
💾 Saving MU model...
  ✓ Model saved to 'mu_model.pt'
```

---

## 🚀 Running Interactive Mode

```bash
python test_mu_model.py
```

### Choose Mode:

**Option 1: Interactive Mode** (recommended)
- Type prompts interactively
- Adjust temperature and length on the fly
- Great for experimentation

**Option 2: Demo Mode**
- Runs predefined prompts
- Shows batch generation
- Good for quick testing

---

## 🎯 Interactive Mode Usage

### Basic Generation:
```
📝 Enter prompt: The quick brown
🤖 Generating (temp=0.8, len=200)...

GENERATED TEXT:
The quick brown fox jumps over the lazy dog and runs through...
```

### Commands:

**Adjust Temperature:**
```
📝 Enter prompt: temp 0.5
✓ Temperature set to 0.5
```

**Adjust Length:**
```
📝 Enter prompt: length 100
✓ Max length set to 100
```

**Quit:**
```
📝 Enter prompt: quit
👋 Goodbye!
```

---

## ⚙️ Generation Parameters

### Temperature
- **Low (0.3-0.6)**: More focused, deterministic
- **Medium (0.7-0.9)**: Balanced (default: 0.8)
- **High (1.0+)**: More creative, random

### Max Length
- **Default**: 200 characters
- **Range**: 50-500 recommended
- Controls how much text to generate

### Top-K Sampling
- **Default**: 40
- Samples from top-k most likely tokens
- Higher = more diverse, lower = more focused

---

## 📝 Example Session

```bash
$ python test_mu_model.py

Using device: cuda

Loading model from mu_model.pt...
✓ Model loaded successfully!
  • Parameters: 6,226,336
  • Vocab size: 10000

================================================================================
Choose mode:
  1. Interactive mode (type prompts)
  2. Demo with sample prompts
================================================================================
Enter choice (1 or 2): 1

================================================================================
🎮 INTERACTIVE MU TRANSFORMER
================================================================================

Commands:
  • Type text to generate continuation
  • 'temp X' to set temperature (e.g., 'temp 0.5')
  • 'length X' to set generation length (e.g., 'length 100')
  • 'quit' or 'exit' to quit

Settings:
  • Temperature: 0.8
  • Max length: 200
  • Top-k: 40
================================================================================

📝 Enter prompt: Once upon a time
🤖 Generating (temp=0.8, len=200)...

================================================================================
GENERATED TEXT:
================================================================================
Once upon a time there was a small village in the mountains where people...
================================================================================

📝 Enter prompt: temp 0.5

✓ Temperature set to 0.5

📝 Enter prompt: The meaning of life
🤖 Generating (temp=0.5, len=200)...

================================================================================
GENERATED TEXT:
================================================================================
The meaning of life is to find purpose in the journey and learn from...
================================================================================

📝 Enter prompt: quit

👋 Goodbye!
```

---

## 🔍 What to Expect

Since the model is trained on **WikiText-2** (character-level):

### ✅ Model Will:
- Generate English text
- Follow basic grammar patterns
- Create coherent character sequences
- Show learned patterns from Wikipedia

### ⚠️ Model Might:
- Generate repetitive text (only 3 epochs)
- Make occasional errors
- Need lower temperature for better quality
- Work best with short prompts

### 💡 Tips for Best Results:
1. **Start with short prompts** (5-15 characters)
2. **Use lower temperature** (0.5-0.7) for coherent text
3. **Try familiar patterns** (e.g., "The ", "In the ")
4. **Experiment with length** - shorter often better
5. **Character-level** - don't expect perfect words

---

## 🎨 Advanced: Programmatic Usage

```python
from test_mu_model import load_model, generate_text

# Load model
model, config, char_to_idx, idx_to_char = load_model('mu_model.pt')

# Generate
text = generate_text(
    model,
    prompt="Hello world",
    char_to_idx=char_to_idx,
    idx_to_char=idx_to_char,
    max_length=150,
    temperature=0.7,
    top_k=40
)

print(text)
```

---

## 🐛 Troubleshooting

### Error: `mu_model.pt not found`
**Solution**: Run training first
```bash
python run_colab.py
```

### Error: `Model doesn't have vocabulary mapping`
**Solution**: Retrain with updated code
```bash
rm mu_model.pt
python run_colab.py
```

### Poor Quality Output
**Try:**
- Lower temperature (0.3-0.5)
- Shorter generation length
- Different prompts
- Train for more epochs

### Out of Memory
**Try:**
- Shorter prompts
- Shorter max_length
- Use CPU instead of CUDA

---

## 📊 Model Info

**Architecture:**
- Dynamic MU Transformer
- 6.2M parameters
- 4×4 semantic slots (I, S, C, R, T, K, G)
- Formula-based dynamic sensitivity
- Character-level generation

**Performance:**
- Val Accuracy: 99.48%
- Val Perplexity: 1.02
- Trained on WikiText-2

---

## 🎯 Next Steps

1. **Test the model** with various prompts
2. **Adjust parameters** to find best settings
3. **Train longer** (10-20 epochs) for better quality
4. **Try different datasets** for other domains
5. **Analyze learned patterns** in semantic slots

---

Enjoy testing your MU Transformer! 🚀
