import torch
import torch.nn as nn
from loss.MATLAB import imresize

def ZeroCenteredGradientPenalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True, only_inputs=True)
    return 0.5 * Gradient.square().sum([1,2,3]).mean()

def RelativisticLoss(PositiveCritics, NegativeCritics):
    return nn.functional.binary_cross_entropy_with_logits(PositiveCritics - NegativeCritics, torch.ones_like(PositiveCritics))

def FidelityLoss(HighResolutionSamples, LowResolutionSamples):
    Downsampled = imresize(HighResolutionSamples, sizes=(LowResolutionSamples.shape[2], LowResolutionSamples.shape[3]))
    return nn.functional.l1_loss(Downsampled, LowResolutionSamples)




### quick test ###
# HighResolutionSamples = torch.randn(10, 3, 192, 192, requires_grad=True)
# LowResolutionSamples = torch.randn(10, 3, 48, 48)
# FidelityLoss(HighResolutionSamples, LowResolutionSamples).backward()