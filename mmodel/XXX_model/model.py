from functools import partial

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mtrain.partial_lr_scheduler import pStep_lr
from mtrain.partial_optimzer import pSGD, pAdam

from ..basic_model import TrainableModel
from .params import params
from .loss import AleatoricCrossEntropyLoss

from torch.nn import functional as F

from mtrain.measurer import RocAucScoreMeasurer


class XXX_model(TrainableModel):
    def __init__(self):
        super().__init__(params)
        # the loss functions needed for training
        self.CEL = torch.nn.CrossEntropyLoss(reduction='sum')
        self.ACEL = AleatoricCrossEntropyLoss(
            monte_carlo_simuls=self.cfg.monte_carlo_simuls)
        # the mectic measurement meter for evaling
        self.measures = [
            RocAucScoreMeasurer(),
        ]

    def prepare_dataloaders(self):
        from .odd_dataset import cifar10_dset, lsun_dset, imagenet_dset, make_odd_dset
        # get training dataset
        cifar_train, cifar_eval = cifar10_dset()
        cls_num = len(cifar_train.classes)  # in this case, 10
        # construct eval datasets for odd
        cifar_lsun_eval = make_odd_dset(cifar_eval, lsun_dset())
        cifar_imgn_eval = make_odd_dset(cifar_eval, imagenet_dset())
        # constructing dataloaders for training dset and eval dset
        _DataLoader = partial(DataLoader, batch_size=self.cfg.batch_size, shuffle=True,
                     num_workers=4, pin_memory=True)
        train_loader = _DataLoader(cifar_train)
        eval_loaders = {
            'cifar_lsun': _DataLoader(cifar_lsun_eval),
            'cifar_imgn': _DataLoader(cifar_imgn_eval)
        }
        return cls_num, train_loader, eval_loaders

    def regist_networks(self):
        from .networks.network import Net
        return {
            "Net":
                Net(num_classes=self.cls_num, depth=self.cfg.depth,
                    widen_factor=self.cfg.widen_factor,
                    dropRate=self.cfg.drop_rate),
        }

    def train_process(self, data, ctx):
        # retrieve training data from train_loader
        img, trg = data

        # forward passing
        logit, var = self.Net(img)

        # calculate lossses
        L_nll = self.CEL(logit, trg)
        L_reg = (self.Net.classifier.logprob_posterior -
                 self.Net.classifier.logprob_prior) / img.shape[0]

        L_gce, L_var, _, L_var_depressor = self.ACEL(var, logit, trg)
        L_aleatoric_uncertainty = L_gce + L_var + L_var_depressor

        L = L_nll + L_reg + L_gce + L_var
        L = L + L_var_depressor if self.current_epoch > 1 else L

        # define optimize config
        # optimizers are defined in /mtrain/partial_optimzer.py
        # lr_schedulers are defined in /mtrain/partial_lr_scheduler.py
        with self.optimize_config(
                optimer=pAdam(lr=1e-3),
                lr_scheduler=pStep_lr(step_size=10, gamma=0.5),
        ):
            self.optimize_loss('global_loss', L, ['Net'])
            self.record_metric('L_nll', L_nll)
            self.record_metric('L_reg', L_reg)
            self.record_metric('L_gce', L_gce)
            self.record_metric('L_var', L_var)

    def eval_process(self, data, ctx):
        # retrieve training data from train_loader
        img, trg = data
        # sample n times, then avg prediction results
        preds_dis = []
        for _ in range(self.cfg.eval_sample_num):
            logit = self.Net(img)[0]
            pred_dis = F.softmax(logit, dim=1)
            preds_dis.append(pred_dis)
        preds_dis = torch.stack(preds_dis, dim=1)
        avg_preds_dis = torch.mean(preds_dis, dim=1)
        return avg_preds_dis, trg
