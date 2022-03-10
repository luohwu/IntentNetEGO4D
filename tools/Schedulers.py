
import torch
import torchvision
import warnings
import matplotlib.pyplot as plt
import  math

class CosExpoScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, switch_step,eta_min=0, gamma=0.995,min_lr=1e-6, last_epoch=-1, verbose=False):

        self.switch_step = switch_step
        self.gamma = gamma
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.min_lr=min_lr
        super(CosExpoScheduler, self).__init__(optimizer, last_epoch, verbose)


    def get_lr(self):

        self.T_cur+=1
        if self.T_cur>self.switch_step:
            return [max(group['lr'] * self.gamma,self.min_lr) for group in self.optimizer.param_groups]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.switch_step)) / 2
                for base_lr in self.base_lrs]


class DecayCosinWarmRestars(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False,decay_rate=1):
        self.decay_rate=decay_rate
        super(DecayCosinWarmRestars,self).__init__(optimizer,T_0,T_mult,eta_min,last_epoch,verbose)

    def step(self, epoch=None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
                # self.eta_min = self.eta_min *self.decay_rate
                self.base_lrs=[max(self.eta_min,p *self.decay_rate) for p in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):


    def __init__(self, optimizer, gamma, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(MinimumExponentialLR, self).__init__(optimizer, gamma,last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [max(group['lr'] * self.gamma,1e-6)
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch,1e-6)
                for base_lr in self.base_lrs]



if __name__=='__main__':
    model=torchvision.models.resnet18(pretrained=False)
    epochs = 200
    ################################################################################
    # optimizer = torch.optim.SGD(models.parameters(), lr=1e-4, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=4e-5, verbose=False)
    # lrs = []
    # for epoch in range(epochs):
    #     lrs.append(optimizer.param_groups[0]['lr'])
    #     scheduler.step()
    # plt.plot(lrs)

    ################################################################################
    # optimizer = torch.optim.SGD(models.parameters(), lr=1e-4, momentum=0.9)
    # scheduler=CosExpoScheduler(optimizer, switch_step=100, eta_min=4e-5, verbose=False)
    # lrs = []
    # for epoch in range(epochs):
    #     lrs.append(optimizer.param_groups[0]['lr'])
    #     scheduler.step()
    # print(f'last lr: {lrs[-1]}')
    # plt.plot(lrs)

    ################################################################################
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
    scheduler=DecayCosinWarmRestars(optimizer,T_0=200,T_mult=2,eta_min=4e-5,decay_rate=0.5,verbose=True)
    lrs = []
    for epoch in range(epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f'last lr: {lrs[-1]}')
    plt.plot(lrs)
    ################################################################################
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-4, momentum=0.9)
    scheduler = DecayCosinWarmRestars(optimizer, T_0=1200, T_mult=2, eta_min=4e-5, decay_rate=0.5,verbose=True)
    lrs = []
    for epoch in range(epochs):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f'last lr: {lrs[-1]}')
    plt.plot(lrs)

    plt.show()