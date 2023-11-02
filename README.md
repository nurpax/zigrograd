# Zigrograd -- scalar-valued autograd written in Zig

A small auto-diffing neural network example written in Zig.  Inspired by [micrograd](https://github.com/karpathy/micrograd).

It's a toy, written just for learning purposes!  For a CPU-only scalar-valued autograd engine, it's fast, but it's no match for real tensor-based frameworks.

## Example

The [src/main.zig](src/main.zig) file implements an MLP model used to classify hand-drawn digits.  It achieves roughly 96% accuracy after training.

A reference implementation of pretty much the same thing can be found in [pytorch/mnist.py](pytorch/mnist.py).

## How to run it

1. Initial setup: `python download_mnist.py` to get training data
2. Start training: `zig build run -Doptimize=ReleaseFast`

Most of the time you want to be running in ReleaseFast mode, as the default debug build is a lot slower.
