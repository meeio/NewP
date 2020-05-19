from torch import nn
from torch.autograd import Function


class GradReverseLayer(nn.Module):

    class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, back_coeff):
        ctx.back_coeff = back_coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        back_coeff = ctx.back_coeff
        reverse_with_coeff = -grad_output*back_coeff
        return reverse_with_coeff, None
        
    def __init__(self, coeff_fn):
        super().__init__()
        self.coeff_fn = coeff_fn
    
    def forward(self, x):
        x = GradReverse.apply(x, self.coeff_fn())
        return x
