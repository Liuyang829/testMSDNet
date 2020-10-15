import torch

for p in range(1, 40):
    print("*********************")
    # 生成一个0.05-1.95
    _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
    print(_p)
    probs = torch.exp(torch.log(_p) * torch.range(1, 7))
    print(probs)
    probs /= probs.sum()
    print(probs)