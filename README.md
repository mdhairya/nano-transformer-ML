# Nano-Transformer-PyTorch

A clean, highly modular, and heavily typed implementation of the original Transformer architecture (*Attention Is All You Need*) built entirely from scratch in pure PyTorch.

This repository prioritizes readability, strict typing, and separation of concerns—designed to demonstrate the exact tensor mechanics of Multi-Head Self-Attention, Positional Encoding, and the Encoder-Decoder stack without relying on high-level library abstractions.

## Features
- **Zero Black Boxes:** Built using only fundamental `torch` operations.
- **Type Hinted:** Full Python static typing for cleaner intelligence and easier debugging.
- **Modular:** Layers, attention mechanisms, and the main model are isolated logically.
- **Device Agnostic:** Auto-detects and runs on CUDA, Apple Silicon (MPS), or CPU.

## Installation

```bash
git clone [https://github.com/yourusername/nano-transformer.git](https://github.com/yourusername/nano-transformer.git)
cd nano-transformer
pip install -r requirements.txt
