import utility
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.NetworksV2 import Discriminator
from loss.LossV2 import ZeroCenteredGradientPenalty, RelativisticLoss

class Adversarial(nn.Module):
    def __init__(self, args, G):
        super(Adversarial, self).__init__()
        
        self.dis = Discriminator(BasisSize=3, StageWidths=[256, 256, 512, 512, 512, 1024], BlocksPerStage=[1, 1, 2, 2, 2, 1])
        
        args.betas = (0, 0.99)
        self.optimizer = utility.make_optimizer(args, self.dis)
        
        self.r1_gamma = args.r1_gamma
        
        self.G = G
        
    def forward(self, fake, real):
        self.dis.requires_grad = True
        self.G.requires_grad = False
            
        self.dis.zero_grad()

        real.requires_grad = True

        output_r = self.dis(real)
        output_f = self.dis(fake.detach())
        
        r1_penalty = ZeroCenteredGradientPenalty(real, output_r)
        
        errD = RelativisticLoss(output_r, output_f) + self.r1_gamma * r1_penalty

        errD.backward()
        self.optimizer.step()

        ###########################
            
        self.dis.requires_grad = False
        self.G.requires_grad = True

        output_f = self.dis(fake)
        output_r = self.dis(real)

        errG = RelativisticLoss(output_f, output_r)
        
        return errG
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)