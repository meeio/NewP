import torch
from abc import ABC, abstractmethod
from sklearn import metrics


class Measurer(ABC):
    def __init__(self, tag):
        self.tag = tag

    @abstractmethod
    def cal(self, preds, targs, ctx):
        pass

class AcuuMeasurer(Measurer):
    def __init__(self, tag='accu'):
        super().__init__(tag)
        self.type = 'scalar'
    
    def cal(self, preds_dis, targs, ctx):
        preds = torch.max(preds_dis, dim=1)[1]
        return preds.eq(targs).float().mean()     

class RocAucScoreMeasurer(Measurer):
    def __init__(self, tag='RocAucScore'):
        super().__init__(tag)
        self.type = 'scalar'
    
    def cal(self, preds_dis, targs, ctx):
        max_probs = torch.max(preds_dis, dim=1)[0]
        max_probs = max_probs.cpu().numpy()
        targs = targs.cpu().numpy()
        ras = metrics.roc_auc_score(targs, max_probs)   
        return torch.tensor(ras)

