import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from itertools import repeat
from pathlib import Path

import torch

from mdata.data_iter import inf_iter
from mtrain import Logger, ObservableTensor, Trainer
from mtrain.logger_writer import CometWriter, WandBWriter
from mtrain.measurer import AcuuMeasurer
from mtrain.utility import anpai


class NoUpdateDict(dict):
    def __setitem__(self, key, value):
        if key in self.keys():
            raise Exception("{} aleady exist".format(key))
        else:
            super().__setitem__(key, value)

class Context(dict):
    def __getitem__(self, key):
        if key not in self.keys():
            super().__setitem__(key, list())
        return super().__getitem__(key)

class Flag:
    END_EPOCH = "end_epoch"
    TRAIN = "train"
    EVAL = "eval"
    EVAL_ACCU = "eval_accuracy"


class optimize_ops:
    def __init__(self, target, optimer, lr_scheduler=None,
                 scheduler_step_signal='epoch'):
        if not isinstance(optimer, partial):
            raise Exception("Use partial for optimer")
        if lr_scheduler is not None and not isinstance(lr_scheduler, partial):
            raise Exception("Use partial for optimer")
        if scheduler_step_signal not in ["epoch", "step"]:
            raise Exception("Must in epoch or step")
        self.optimer_fn = optimer
        self.lr_scheduler_fn = lr_scheduler
        self.scheduler_step_signal = scheduler_step_signal
        self.target = target

    def __enter__(self):
        self.target.optimer_fn = self.optimer_fn
        self.target.lr_scheduler_fn = self.lr_scheduler_fn
        self.target.scheduler_step_signal = self.scheduler_step_signal

    def __exit__(self, *args):
        self.target.optimer_fn = None
        self.target.lr_scheduler_fn = None
        self.target.lr_scheduler_signal = None


class TrainableModel(ABC):
    def __init__(self, cfg):

        super(TrainableModel, self).__init__()

        target_model = os.environ['TARGET_MODEL']
        model_path = Path(__file__).parent.joinpath(target_model,
                                                    '.saved_model')
        model_path.mkdir(exist_ok=True)

        writer = WandBWriter(cfg.project_name, cfg.enable_log, cfg)
        writer.log_params(cfg)
        self.cfg = cfg
        self.writer = writer

        # get class infos and dataloaders
        cls_num, train_dset, eval_dset = self.prepare_dataloaders()
        is_dict_train_dest = isinstance(train_dset, dict)
        if is_dict_train_dest:
            dset_items = list(train_dset.items())
            train_dset_key, train_dset = dset_items[0]
            other_dset_its = {k: inf_iter(d) for k, d in dset_items[1:]}

        is_dict_eval_dset = isinstance(eval_dset, dict)
        if not is_dict_eval_dset:
            eval_dset = {'dataset': eval_dset}

        # data feeding function
        def iter_dsets(mode):
            if mode == Flag.TRAIN:
                self._current_dset = 'train'
                for data in iter(train_dset):
                    if is_dict_train_dest:
                        _data = dict()
                        _data[train_dset_key] = anpai(data)
                        for key, it in other_dset_its:
                            _data[key] = anpai(next(it))
                        data = _data
                    else:
                        data = anpai(data)
                    yield data
            elif mode == Flag.EVAL:
                for key in eval_dset:
                    self._current_dset = key
                    yield key, eval_dset[key]

        self.iter_dsets = iter_dsets
        self.confusion_matrix = {
            k: torch.zeros(cls_num, cls_num)
            for k in eval_dset
        }
        self._best_eval_accu = {k: 0 for k in eval_dset}
        self.cls_num = cls_num

        # get all networks and send networks to gup
        networks = self.regist_networks()
        assert isinstance(networks, dict)
        networks = {i: anpai(j) for i, j in networks.items()}
        # make network to be class attrs
        for i, j in networks.items():
            self.__setattr__(i, j)
        self.networks = networks

        self.current_step = 0
        self.current_epoch = 0
        self.epoch_steps = len(train_dset)
        self.tensors = NoUpdateDict()
        self.trainer = NoUpdateDict()
        self.loggers = NoUpdateDict()
        self.optimize_config = partial(optimize_ops, target=self)

        self.measures = []
        self.measures.append(AcuuMeasurer())

        # testing
        self._current_mode = ''

    @abstractmethod
    def prepare_dataloaders(self):
        pass

    @abstractmethod
    def regist_networks(self):
        pass

    @abstractmethod
    def train_process(self, data, ctx):
        pass

    @abstractmethod
    def eval_process(self, data, ctx):
        pass

    def train_model(self):
        for i in range(self.cfg.epoch):
            # begin training in current epoch
            self._current_mode = Flag.TRAIN
            for _, i in self.networks.items():
                i.train(True)

            ctx = Context()
            for datas in self.iter_dsets(mode=Flag.TRAIN):
                self.train_process(datas, ctx)
                self.current_step += 1
                self.current_epoch = self.current_step / self.epoch_steps
            assert self.current_epoch == int(self.current_epoch)

            # begin eval if nedded
            if self.current_epoch % self.cfg.eval_epoch_interval == 0:
                self.eval_model()

    def eval_model(self, **kwargs):
        # set all networks to eval mode
        self._current_mode = Flag.EVAL
        for _, i in self.networks.items():
            i.eval()

        # iter every eval dataset
        with torch.no_grad():
            for key, dset in self.iter_dsets(mode=Flag.EVAL):
                ctx = Context()
                ctx.dataset = key
                preds = []
                targs = []
                for data in iter(dset):
                    data = anpai(data)
                    pred, targ = self.eval_process(data, ctx)
                    preds.append(pred)
                    targs.append(targ)
                ctx = {k: torch.cat(ctx[k], dim=0) for k in ctx}
                preds = torch.cat(preds, dim=0)
                targs = torch.cat(targs, dim=0)
                for met in self.measures:
                    tag = met.tag + '@eval_' + key
                    val = met.cal(preds, targs, ctx)
                    self.record_metric(tag, val, met.type)

    def optimize_loss(self, name, value, networks, lr_mult=1):
        if name not in self.tensors:
            assert isinstance(networks, (tuple, list))
            networks = {k: self.networks[k] for k in networks}

            optimer_fn = self.optimer_fn
            lr_scheduler_fn = self.lr_scheduler_fn
            scheduler_step_signal = self.scheduler_step_signal
            if optimer_fn is None:
                raise Exception("need to set optiemr with 'with'.")

            loss = ObservableTensor(name)
            trainer = Trainer(networks, optimer_fn, lr_scheduler_fn,
                              scheduler_step_signal, lr_mult)
            logger = Logger(name, self.cfg.log_step_interval, self.writer)
            loss.add_listener([trainer, logger])

            self.tensors[name] = loss
            self.trainer[name] = trainer
            self.loggers[name] = logger

        self.tensors[name].update(value, self.current_step, self.current_epoch)

    def record_metric(self, name, value, ttype='scalar'):
        if name not in self.tensors:
            tensor = ObservableTensor(name)
            log_interval = 1 if self._current_mode == Flag.EVAL else self.cfg.log_step_interval
            logger = Logger(name, log_interval, self.writer, ttype=ttype)
            tensor.add_listener(logger)
            self.tensors[name] = tensor
            self.loggers[name] = logger
        self.tensors[name].update(value, self.current_step, self.current_epoch)
