# BVH, Jun 2022.

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'eval/'))

from __init__ import *


class MyPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''
    def __init__(self):
        super().__init__()
        self.state = [100, 200, 300, 400]

    def forward(self, inds):
        within_batch_idx = inds[0].item()
        time.sleep(within_batch_idx / 50.0)
        self.state[within_batch_idx] += 1
        print(inds, self.state)
        return inds / 10
        # for i in range(4):
        #     print(i, x.shape, x)
        #     time.sleep(max(x[0].item() / 2.0 + 1.0, 0.5))
        # y = torch.square(x)
        # return y


def main():
    
    device = torch.device('cuda')
    net = MyPipeline()
    net = net.to(device)
    net = torch.nn.DataParallel(net, device_ids=[0, 0, 1, 1])

    # x = torch.randn(2, 1)
    # y = net(x)

    # x = torch.randn(3, 1)
    # y = net(x)

    # x = torch.randn(4, 1)
    # y = net(x)

    inds = torch.arange(0, 4, dtype=torch.int32).to(device)
    y = net(inds)
    time.sleep(0.5)
    y = net(inds)
    time.sleep(0.5)
    y = net(inds)
    time.sleep(0.5)
    y = net(inds)
    time.sleep(0.5)
    y = net(inds)
    time.sleep(0.5)
    y = net(inds)

    pass


if __name__ == '__main__':

    # DEBUG / WARNING: This is slow, but we can detect NaNs this way:
    # torch.autograd.set_detect_anomaly(True)

    np.set_printoptions(precision=3, suppress=True)
    torch.set_printoptions(precision=3, sci_mode=False)

    main()
