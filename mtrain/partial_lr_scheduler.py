from torch.optim import lr_scheduler
from functools import partial as P

def pStep_lr(step_size, gamma=0.1, last_epoch=-1):
    return P(lr_scheduler.StepLR, step_size=step_size, gamma=gamma, last_epoch=last_epoch)
