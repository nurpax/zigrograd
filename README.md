# Zigrograd -- scalar-valued autograd written in Zig

A small auto-diffing neural network example written in Zig.  Inspired by [micrograd](https://github.com/karpathy/micrograd).

## Example

The [src/main.zig](src/main.zig) file implements an MLP model used to classify hand-drawn digits.  It achieves roughly 96% accuracy after training.

A reference implementation of pretty much the same thing can be found in [pytorch/mnist.py](pytorch/mnist.py).
