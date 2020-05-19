import torch
import torch.nn as nn
import torch.nn.functional as F

from .bayes import bLinear
from .wideresnet import WideResNet


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]
    [0]: https://arxiv.org/abs/1704.03162
    """
    def __init__(self, num_classes, depth=40, widen_factor=4, dropRate=0.0):
        super(Net, self).__init__()

        self.feature_extractor = WideResNet(depth=depth,
                                            num_classes=num_classes,
                                            widen_factor=widen_factor,
                                            dropRate=dropRate)

        self.classifier = bLinear(64 * widen_factor, num_classes)
        self.logits_variance = nn.Linear(num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
 
    def forward(self, input, sample=True):
        f = self.feature_extractor(input)

        # a = self.attention(f)

        # w_f, at_prob = apply_attention(f, a)

        logit_pred = self.classifier(f, sample)
        logit_var = self.logits_variance(logit_pred)
        var = F.softplus(logit_var)
        return logit_pred, var

    def neg_elbo(self, inputs, target):

        # prob_test_out = torch.zeros(self.sample, x.shape[0], 10).to(DEVICE)
        # for i in range(self.sample):
        #    prob_test_out[i] = self(x, sample_flag=True)
        # logits = prob_test_out.mean(0)

        # nll = self.criterion(logits, target)\

        logits = self(inputs)
        nll = self.criterion(logits, target)
        regularizer = (self.classifier.logprob_posterior -
                       self.classifier.logprob_prior) / inputs.shape[0]

        loss = nll + regularizer

        # loss,logits,nll,regularizer=self.feature_extractor.neg_elbo(input,target)
        # print("nll",nll)
        # print("regularizer",regularize)
        # loss = nll+ regularizer
        return loss, logits, nll, regularizer


# class Uncertainty(nn.Module):
#     def __init__(self, out_features):
#         super(Uncertainty, self).__init__()
#         self.logits_variance_linear = nn.Linear(out_features, 1)

#     def forward(self, logit):
#         variance = F.softplus(self.logits_variance_linear(logit))
#         return variance



