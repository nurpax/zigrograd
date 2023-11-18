# Zigrograd -- scalar-valued autograd written in Zig

A small auto-diffing neural network example written in Zig.  Heavily inspired by [micrograd](https://github.com/karpathy/micrograd).  The original scalar-valued version can be found under the [scalar](https://github.com/nurpax/zigrograd/tree/scalar) tag.  The version in `main` is vector-valued using a mini-numpy implemented in [src/ndarray.zig](src/ndarray.zig)

It's a toy, written just for learning purposes!  It's pretty fast for what it is but no match for real tensor-based frameworks.

## Example

The [src/main.zig](src/main.zig) file implements an MLP model used to classify hand-drawn digits.  It achieves roughly 96% accuracy after training.

A reference implementation of pretty much the same thing can be found in [pytorch/mnist.py](pytorch/mnist.py).

## How to run it

1. Initial setup: `python download_mnist.py` to download the MNIST dataset
2. Start training: `zig build run -Doptimize=ReleaseFast`

Most of the time you want to be running in ReleaseFast mode, as the default debug build is a lot slower.
