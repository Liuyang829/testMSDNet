import torch
pthfile = r'/data1/master1/MSDNet-PyTorch/cifar100_anytime_result/flops.pth'
net = torch.load(pthfile)
print(net)