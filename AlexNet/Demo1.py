import torch

if __name__ == '__main__':
    a = torch.randn(3, 1, 2)
    print(a)
    _,index = a.max(2)
    print(_.shape)
    print(index.shape)
    print(_)
    print(index)
    # print(a.expand(3, 3, 2))
