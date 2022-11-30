import torch

net=torch.load(r'./msssim_l1_0.84/model_state/50_epoch.pth')
for k in net.keys():
    print(k)



