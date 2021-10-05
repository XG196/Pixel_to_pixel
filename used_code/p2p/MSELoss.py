import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        # print("PLCC input: " + str(input) + " " + str(target)
        n = 1
        for sizes in input.shape:
            n *= sizes 
        self.loss = torch.sum((input - target) ** 2) / n
        return self.loss

class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        self.loss = torch.mean(torch.abs(input - target))
        return self.loss


# PLCC unittest
if __name__ == '__main__':
    mse_loss = MSELoss()
    test_input = Variable(torch.tensor([-3.9686, -2.0431]), requires_grad=True)
    test_target = Variable(torch.tensor([0.9124, 0.9973]))
    optimizer = optim.SGD([test_input], lr=1)

    while True:
        mse = mse_loss(test_input, test_target)
        print(mse.item())
        mse.backward()
        optimizer.step()
