"""
LSTM Text Generator - Complete Implementation
Combines model, training, and generation in a single file.
"""

import argparse
import os
import pickle
import urllib.request
import urllib.error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class LSTMTexGenerator(nn.Module):
    """
    Character-level LSTM model for text generation.
    
    Args:
        vocab_size: Size of the vocabulary (number of unique characters)
        embedding_dim: Dimension of character embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, 
                 num_layers=2, dropout=0.2):
        super(LSTMTexGenerator, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer: converts character indices to dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers: processes sequences
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer: maps LSTM output to vocabulary logits
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
            hidden: Tuple of (h_n, c_n) hidden states from previous step
        
        Returns:
            output: Logits of shape (batch_size, seq_length, vocab_size)
            hidden: Updated hidden states
        """
        # Embed characters
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # Pass through LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Map to vocabulary
        output = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        """
        Initialize hidden states for the LSTM.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on
        
        Returns:
            Tuple of (h_0, c_0) initialized hidden states
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)
    
    def generate(self, char_to_idx, idx_to_char, seed_text, length=100, 
                 temperature=1.0, device='cpu'):
        """
        Generate text using the trained model.
        
        Args:
            char_to_idx: Dictionary mapping characters to indices
            idx_to_char: Dictionary mapping indices to characters
            seed_text: Starting text for generation
            length: Number of characters to generate
            temperature: Controls randomness (higher = more random)
            device: Device to run inference on
        
        Returns:
            Generated text string
        """
        self.eval()
        generated = seed_text
        
        # Initialize hidden state
        hidden = self.init_hidden(1, device)
        
        # Process seed text to get initial hidden state
        with torch.no_grad():
            for char in seed_text:
                if char in char_to_idx:
                    char_idx = torch.tensor([[char_to_idx[char]]]).to(device)
                    _, hidden = self.forward(char_idx, hidden)
            
            # Generate new characters
            for _ in range(length):
                # Get last character
                last_char = generated[-1]
                if last_char not in char_to_idx:
                    last_char = list(char_to_idx.keys())[0]  # Fallback
                
                char_idx = torch.tensor([[char_to_idx[last_char]]]).to(device)
                
                # Forward pass
                output, hidden = self.forward(char_idx, hidden)
                
                # Apply temperature and get probabilities
                logits = output[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Sample next character
                char_idx = torch.multinomial(probs, 1).item()
                generated += idx_to_char[char_idx]
        
        return generated


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_text_file(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_vocab(text):
    """Create vocabulary mappings from text."""
    # Get unique characters
    chars = sorted(list(set(text)))
    
    # Create mappings
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    vocab_size = len(chars)
    
    return char_to_idx, idx_to_char, vocab_size


def encode_text(text, char_to_idx):
    """Encode text to integer indices."""
    return [char_to_idx[char] for char in text]


def decode_text(encoded, idx_to_char):
    """Decode integer indices to text."""
    return ''.join([idx_to_char[idx] for idx in encoded])


def create_sequences(text_encoded, seq_length=100):
    """Create sequences for training."""
    X = []
    y = []
    
    for i in range(len(text_encoded) - seq_length):
        X.append(text_encoded[i:i + seq_length])
        y.append(text_encoded[i + 1:i + seq_length + 1])
    
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def save_model(model, filepath):
    """Save model to file."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'embedding_dim': model.embedding_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
    }, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath, device='cpu'):
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location=device)
    
    model = LSTMTexGenerator(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


# ============================================================================
# DATASET CLASS
# ============================================================================

class TextDataset(Dataset):
    """Dataset class for text sequences."""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X, y in tqdm(dataloader, desc="Training"):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output, _ = model(X)
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        y = y.reshape(-1)
        
        # Calculate loss
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model(args):
    """Main training function."""
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    text = load_text_file(args.data)
    print(f"Text length: {len(text)} characters")
    
    # Create vocabulary
    char_to_idx, idx_to_char, vocab_size = create_vocab(text)
    print(f"Vocabulary size: {vocab_size}")
    
    # Save vocabulary for later use
    vocab_path = os.path.join(args.save_dir, 'vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump((char_to_idx, idx_to_char), f)
    print(f"Vocabulary saved to {vocab_path}")
    
    # Encode text
    text_encoded = encode_text(text, char_to_idx)
    
    # Create sequences
    print("Creating sequences...")
    X, y = create_sequences(text_encoded, args.seq_length)
    print(f"Created {len(X)} sequences")
    
    # Create dataset and dataloader
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMTexGenerator(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("\nStarting training...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        loss = train_epoch(model, dataloader, criterion, optimizer, device)
        
        print(f"Loss: {loss:.4f}")
        
        # Update learning rate
        scheduler.step(loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_model(model, checkpoint_path)
        
        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.save_dir, 'best_model.pth')
            save_model(model, best_path)
            print(f"New best model saved! Loss: {loss:.4f}")
    
    print("\nTraining completed!")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


# ============================================================================
# DATASET DOWNLOAD FUNCTIONS
# ============================================================================

def download_dataset(url, output_file, max_chars=None):
    """
    Download text dataset from URL and save to file.
    
    Args:
        url: URL to download from
        output_file: Path to save the text file
        max_chars: Maximum characters to save (None = save all)
    """
    print(f"Downloading dataset from: {url}")
    print(f"Output file: {output_file}")
    
    try:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Download the file with chunked reading to handle large files
        print("Downloading...")
        text = ""
        chunk_size = 8192  # 8KB chunks
        max_bytes = max_chars if max_chars else float('inf')
        bytes_read = 0
        
        # Create request with headers to avoid blocking
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        with urllib.request.urlopen(req, timeout=30) as response:
            while bytes_read < max_bytes:
                try:
                    chunk = response.read(min(chunk_size, max_bytes - bytes_read))
                    if not chunk:
                        break
                    text += chunk.decode('utf-8', errors='ignore')
                    bytes_read += len(chunk)
                    
                    # Show progress for large files
                    if bytes_read % (chunk_size * 100) == 0:
                        print(f"  Downloaded: {bytes_read:,} bytes...", end='\r')
                except Exception as e:
                    # If we got some data, continue with what we have
                    if text:
                        print(f"\n  Warning: Download incomplete, but got {len(text):,} characters")
                        break
                    else:
                        raise
        
        print()  # New line after progress
        
        # Truncate if max_chars is specified
        if max_chars and len(text) > max_chars:
            print(f"Truncating to {max_chars:,} characters (downloaded: {len(text):,})")
            text = text[:max_chars]
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ Successfully downloaded and saved to {output_file}")
        print(f"  File size: {len(text):,} characters")
        return True
        
    except urllib.error.URLError as e:
        print(f"✗ Error downloading from URL: {e}")
        print("  Please check the URL and your internet connection.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_dataset_command(args):
    """Handle dataset download command."""
    success = download_dataset(
        url=args.url,
        output_file=args.output,
        max_chars=args.max_chars
    )
    
    if success:
        print("\n" + "="*60)
        print("Next steps:")
        print("="*60)
        print(f"1. Train the model:")
        print(f"   python lstm_text_generator.py train --data {args.output} --epochs 10")
        print()
        print("2. Generate text:")
        print("   python lstm_text_generator.py generate --model checkpoints/best_model.pth --seed \"The \" --length 500")
        print()


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_text(args):
    """Main generation function."""
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    if args.vocab is None:
        # Try to find vocab in same directory as model
        model_dir = os.path.dirname(args.model)
        vocab_path = os.path.join(model_dir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            args.vocab = vocab_path
        else:
            raise ValueError("Vocabulary file not found. Please specify --vocab")
    
    print(f"Loading vocabulary from {args.vocab}...")
    with open(args.vocab, 'rb') as f:
        char_to_idx, idx_to_char = pickle.load(f)
    
    print(f"Vocabulary size: {len(char_to_idx)}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device)
    model.eval()
    
    print(f"Generating {args.length} characters with temperature {args.temperature}...")
    print(f"Seed text: '{args.seed}'")
    print("-" * 50)
    
    # Generate text
    generated_text = model.generate(
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        seed_text=args.seed,
        length=args.length,
        temperature=args.temperature,
        device=device
    )
    
    # Print generated text
    print(generated_text)
    print("-" * 50)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"\nGenerated text saved to {args.output}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LSTM Text Generator - Train or Generate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download dataset from URL
  python lstm_text_generator.py download --url https://example.com/text.txt --output data/dataset.txt
  
  # Train the model
  python lstm_text_generator.py train --data data/shakespeare_sample.txt --epochs 10
  
  # Generate text
  python lstm_text_generator.py generate --model checkpoints/best_model.pth --seed "The " --length 500
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode: train or generate')
    
    # Training parser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Path to training text file')
    train_parser.add_argument('--seq_length', type=int, default=100,
                             help='Sequence length for training (default: 100)')
    train_parser.add_argument('--embedding_dim', type=int, default=128,
                             help='Embedding dimension (default: 128)')
    train_parser.add_argument('--hidden_dim', type=int, default=512,
                             help='Hidden dimension (default: 512)')
    train_parser.add_argument('--num_layers', type=int, default=2,
                             help='Number of LSTM layers (default: 2)')
    train_parser.add_argument('--dropout', type=float, default=0.2,
                             help='Dropout rate (default: 0.2)')
    train_parser.add_argument('--batch_size', type=int, default=64,
                             help='Batch size (default: 64)')
    train_parser.add_argument('--epochs', type=int, default=20,
                             help='Number of epochs (default: 20)')
    train_parser.add_argument('--learning_rate', type=float, default=0.001,
                             help='Learning rate (default: 0.001)')
    train_parser.add_argument('--save_dir', type=str, default='./checkpoints',
                             help='Directory to save checkpoints (default: ./checkpoints)')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (cpu/cuda/auto, default: auto)')
    
    # Generation parser
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--model', type=str, required=True,
                           help='Path to trained model checkpoint')
    gen_parser.add_argument('--vocab', type=str, default=None,
                           help='Path to vocabulary file (default: same directory as model)')
    gen_parser.add_argument('--seed', type=str, default="The ",
                           help='Seed text to start generation (default: "The ")')
    gen_parser.add_argument('--length', type=int, default=500,
                           help='Number of characters to generate (default: 500)')
    gen_parser.add_argument('--temperature', type=float, default=1.0,
                           help='Temperature for sampling (higher = more random, default: 1.0)')
    gen_parser.add_argument('--device', type=str, default='auto',
                           help='Device to use (cpu/cuda/auto, default: auto)')
    gen_parser.add_argument('--output', type=str, default=None,
                           help='Output file to save generated text (optional)')
    
    # Download parser
    download_parser = subparsers.add_parser('download', help='Download dataset from URL')
    download_parser.add_argument('--url', type=str, required=True,
                                help='URL to download dataset from')
    download_parser.add_argument('--output', type=str, required=True,
                                help='Output file path to save the dataset')
    download_parser.add_argument('--max_chars', type=int, default=None,
                                help='Maximum characters to save (default: save all)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'generate':
        generate_text(args)
    elif args.mode == 'download':
        download_dataset_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

