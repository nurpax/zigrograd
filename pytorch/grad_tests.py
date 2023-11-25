import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def print_f32_slice(tensor: torch.Tensor, name):
    xs = tensor.detach().numpy().flatten()
    print(f'const {name}_val = [_]f32{{{", ".join([str(x) for x in xs])}}};')

def print_tensor_init(tensor: torch.Tensor, name: str):
    shape = ', '.join([str(x) for x in tensor.shape])
    print_f32_slice(tensor, name)
    print(f'var {name}_arr = Ndarray(f32).initFromSlice1d(h.allocator(), &{name}_val);')
    print(f'{name}_arr = {name}_arr.reshape(h.allocator(), &[_]usize{{ {shape} }});')

def conv2d_test():
    x = (0.5+np.arange(14, dtype=np.float32))/13.5
    x = np.stack([x + i for i in range(28)])
    x = np.random.randn(1, 2, 8, 8)

    K = 3
    w = np.random.randn(4, 2, K, K)

    weight = torch.tensor(w, requires_grad=True)
    x = torch.tensor(x, requires_grad=True)

    out = F.conv2d(x, weight)
    #print(out.detach().numpy()[0,0,0])
    #print(out.detach().numpy()[0,0,1])

    loss = out.sum()
    loss.backward()
    print('out shape', out.shape)
    print('dw', weight.grad)
    print('dx', x.grad)

    # print zig test case
    print_tensor_init(x, 'x')
    print_tensor_init(weight, 'w')
    print_f32_slice(out, 'out_expected')
    assert weight.grad is not None and x.grad is not None
    print_f32_slice(weight.grad, 'dw_expected')
    print_f32_slice(x.grad, 'dx_expected')


def avgpool2d_test():
    x = (0.5+np.arange(8, dtype=np.float32))/7.5
    x = np.stack([x + i for i in range(8)])
    x = torch.tensor(np.stack([x]))
    b = torch.ones_like(x, requires_grad=True)
    x = x * b
    x = F.avg_pool2d(x, 2)
    loss = x.sum()
    loss.backward()
    print('avg pool')
    print(x)
    print('b.grad')
    print(b.grad)

def main():
    np.random.seed(0x43534)
    conv2d_test()
    avgpool2d_test()
    return
    x = (0.5+np.arange(28, dtype=np.float32))/27.5
    x = np.stack([x + i for i in range(28)])

    # w = (-1+np.arange(3, dtype=np.float32))
    # w = np.stack([w + i for i in range(3)])
    w = [[1,2,3], [4,5,6], [7,8,9]]
    w = np.array(w, dtype=np.float32)

    x_in = np.stack([[x]])
    w_in = np.stack([[w]]) # N, C_out, C_in, H, W
    weight = torch.tensor(w_in, requires_grad=True)
    out = F.conv2d(torch.tensor(x_in), weight)
    print(out.detach().numpy()[0,0,0])
    print(out.detach().numpy()[0,0,1])

    loss = out.sum()
    loss.backward()
    print(weight.grad)

if __name__ == '__main__':
    main()
