# LSTM Text Generator

A character-level LSTM model for text generation implemented in PyTorch. **Everything is in a single Python file** - easy to use, share, and deploy!

## Features

- **Single File Implementation**: All functionality in one file (`lstm_text_generator.py`)
- **Character-level text generation**: Generates text character by character
- **Flexible architecture**: Configurable embedding dimensions, hidden units, and LSTM layers
- **Temperature-based sampling**: Control the randomness of generated text
- **Dataset download**: Download datasets directly from URLs
- **GPU support**: Automatic GPU detection and usage
- **Checkpoint saving**: Save and load trained models
- **Easy to use**: Simple command-line interface

## Project Structure

```
.
├── lstm_text_generator.py  # Complete implementation (all in one file!)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── SINGLE_FILE_USAGE.md    # Detailed usage guide
├── DOWNLOAD_EXAMPLES.md    # Dataset download examples
└── data/                   # Training data directory
└── checkpoints/            # Saved models directory
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Download Dataset (Optional)

Download a dataset from a URL:

```bash
python lstm_text_generator.py download --url https://www.gutenberg.org/files/100/100-0.txt --output data/shakespeare.txt --max_chars 100000
```

### 2. Train the Model

Train the LSTM model on your text data:

```bash
python lstm_text_generator.py train --data data/shakespeare.txt --epochs 10
```

### 3. Generate Text

Generate text using the trained model:

```bash
python lstm_text_generator.py generate --model checkpoints/best_model.pth --seed "The " --length 500
```

## Usage

### Download Dataset

Download text datasets from URLs:

```bash
python lstm_text_generator.py download --url <URL> --output <output_file> [--max_chars <limit>]
```

**Examples:**
```bash
# Download Shakespeare
python lstm_text_generator.py download --url https://www.gutenberg.org/files/100/100-0.txt --output data/shakespeare.txt

# Download with character limit
python lstm_text_generator.py download --url https://www.gutenberg.org/files/100/100-0.txt --output data/shakespeare.txt --max_chars 100000
```

### Train the Model

```bash
python lstm_text_generator.py train --data <text_file> [options]
```

**Required:**
- `--data`: Path to training text file

**Optional:**
- `--seq_length`: Sequence length for training (default: 100)
- `--embedding_dim`: Embedding dimension (default: 128)
- `--hidden_dim`: Hidden dimension (default: 512)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout rate (default: 0.2)
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 20)
- `--learning_rate`: Learning rate (default: 0.001)
- `--save_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--device`: Device to use (cpu/cuda/auto, default: auto)

**Examples:**

```bash
# Basic training
python lstm_text_generator.py train --data data/shakespeare.txt --epochs 10

# Custom parameters
python lstm_text_generator.py train \
    --data data/shakespeare.txt \
    --epochs 20 \
    --batch_size 128 \
    --hidden_dim 1024 \
    --num_layers 3 \
    --learning_rate 0.0005

# Fast training (smaller model)
python lstm_text_generator.py train \
    --data data/shakespeare.txt \
    --epochs 5 \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_layers 1
```

### Generate Text

```bash
python lstm_text_generator.py generate --model <model_path> [options]
```

**Required:**
- `--model`: Path to trained model checkpoint

**Optional:**
- `--vocab`: Path to vocabulary file (auto-detected if in same directory as model)
- `--seed`: Seed text to start generation (default: "The ")
- `--length`: Number of characters to generate (default: 500)
- `--temperature`: Temperature for sampling (default: 1.0)
  - Lower values (0.1-0.5): More conservative, repetitive text
  - Higher values (1.0-2.0): More creative, diverse text
- `--device`: Device to use (cpu/cuda/auto, default: auto)
- `--output`: Output file to save generated text (optional)

**Examples:**

```bash
# Basic generation
python lstm_text_generator.py generate --model checkpoints/best_model.pth --seed "The " --length 500

# Low temperature (more predictable)
python lstm_text_generator.py generate \
    --model checkpoints/best_model.pth \
    --seed "Once upon a time" \
    --length 1000 \
    --temperature 0.5

# High temperature (more creative)
python lstm_text_generator.py generate \
    --model checkpoints/best_model.pth \
    --seed "In the future" \
    --length 800 \
    --temperature 1.5 \
    --output generated_text.txt
```

## Model Architecture

The model consists of:
1. **Embedding Layer**: Maps characters to dense vectors
2. **LSTM Layers**: Processes sequences with long-term memory
3. **Fully Connected Layer**: Maps LSTM output to vocabulary logits
4. **Dropout**: Regularization to prevent overfitting

## Complete Workflow Example

```bash
# 1. Download dataset
python lstm_text_generator.py download \
    --url https://www.gutenberg.org/files/100/100-0.txt \
    --output data/shakespeare.txt \
    --max_chars 100000

# 2. Train the model
python lstm_text_generator.py train \
    --data data/shakespeare.txt \
    --epochs 10 \
    --batch_size 64

# 3. Generate text
python lstm_text_generator.py generate \
    --model checkpoints/best_model.pth \
    --seed "To be or not to be" \
    --length 500 \
    --temperature 1.0
```

## Tips for Better Results

1. **More data**: Larger training datasets generally produce better results (100k+ characters recommended)
2. **More epochs**: Train for more epochs, but watch for overfitting (20-50 epochs often good)
3. **Larger model**: Increase `hidden_dim` and `num_layers` for more capacity
4. **Learning rate**: Experiment with different learning rates (0.0005-0.002)
5. **Sequence length**: Longer sequences can capture more context (100-200)
6. **Temperature tuning**: Adjust temperature based on desired creativity level

## Popular Dataset Sources

### Project Gutenberg (Free Books)
- Shakespeare: `https://www.gutenberg.org/files/100/100-0.txt`
- Alice in Wonderland: `https://www.gutenberg.org/files/11/11-0.txt`
- Moby Dick: `https://www.gutenberg.org/files/2701/2701-0.txt`

### Other Sources
- Any direct link to a `.txt` file
- GitHub raw files
- Public text datasets

## Troubleshooting

**Out of memory errors:**
- Reduce `--batch_size`
- Reduce `--hidden_dim` or `--num_layers`
- Reduce `--seq_length`

**Poor generation quality:**
- Train for more epochs
- Increase model size (`--hidden_dim`, `--num_layers`)
- Use more training data
- Adjust learning rate

**Slow training:**
- Use GPU if available (set `--device cuda`)
- Increase `--batch_size` if memory allows
- Reduce `--seq_length`
- Use smaller model for faster training

**Download errors:**
- Check your internet connection
- Verify the URL is correct and accessible
- Some websites may block automated downloads
- Try using `--max_chars` to limit file size

## Help

Get help for any command:

```bash
python lstm_text_generator.py --help
python lstm_text_generator.py train --help
python lstm_text_generator.py generate --help
python lstm_text_generator.py download --help
```

## Advantages of Single File

✅ **Everything in one place** - Easy to share and deploy
✅ **No need to manage multiple files** - Simpler project structure
✅ **Same functionality** - All features from separate files
✅ **Cleaner imports** - No cross-file dependencies
✅ **Easy to understand** - All code in one location

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

This implementation uses PyTorch for deep learning. The architecture is based on standard LSTM text generation approaches.
