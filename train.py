import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from nano_transformer import Transformer

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print(f"Initializing Nano-Transformer with {args.num_layers} layers...")
    model = Transformer(
        src_vocab_size=args.vocab_size,
        tgt_vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore padding token
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Dummy Data Generation
    batch_size = 4
    src_data = torch.randint(1, args.vocab_size, (batch_size, 15)).to(device)
    tgt_data = torch.randint(1, args.vocab_size, (batch_size, 15)).to(device)

    # Dummy Training Loop
    model.train()
    print("Starting mock training step...")
    optimizer.zero_grad()
    
    # Input to decoder is shifted right (ignoring last token)
    tgt_input = tgt_data[:, :-1]
    # Expected output is shifted left (predicting next token)
    tgt_expected = tgt_data[:, 1:]

    output = model(src_data, tgt_input)
    
    # Reshape for CrossEntropyLoss
    loss = criterion(output.reshape(-1, args.vocab_size), tgt_expected.reshape(-1))
    
    loss.backward()
    optimizer.step()

    print(f"Forward & Backward pass complete. Initial Loss: {loss.item():.4f}")
    print("Project is ready for a real dataset!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Nano-Transformer")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--d_model", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of encoder/decoder layers")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    args = parser.parse_args()
    main(args)
