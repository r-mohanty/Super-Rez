import utility
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.NetworksV2 import Discriminator
from loss.LossV2 import ZeroCenteredGradientPenalty, RelativisticLoss, FidelityLoss

class Adversarial(nn.Module):
    def __init__(self, args, G, fidelity_weight, regularization_weight):
        super(Adversarial, self).__init__()
        
        self.dis = Discriminator(BasisSize=3, StageWidths=[256, 256, 512, 512, 512, 1024], BlocksPerStage=[1, 1, 2, 2, 2, 1])
        self.optimizer = utility.make_optimizer(args, self.dis, is_dis=True)
        self.r1_gamma = args.r1_gamma
        
        self.G = G
        
        self.fidelity_weight = fidelity_weight
        self.regularization_weight = regularization_weight
        
        self.grad_accum = args.grad_accum
        
        
        
    def eval_D_once(self, lr, real):
        real.requires_grad = True

        output_r = self.dis(real)
        output_f = self.dis(self.G(lr, 0).detach())
        
        adv = RelativisticLoss(output_r, output_f) / self.grad_accum
        r1_penalty = self.r1_gamma * ZeroCenteredGradientPenalty(real, output_r) / self.grad_accum
        
        (adv + r1_penalty).backward()
        
        return adv.detach().item(), r1_penalty.detach().item()
        
        
    def updateD(self, lr, real):
        self.dis.requires_grad = True
        self.G.requires_grad = False
            
        self.dis.zero_grad()
        
        batch_size = lr.shape[0]
        chunk_size = batch_size // self.grad_accum
        
        
        adv_accum = 0
        r1_accum = 0
        
        for x in range(self.grad_accum):
            adv, r1 = self.eval_D_once(lr[x * chunk_size : (x + 1) * chunk_size, :, :, :], real[x * chunk_size : (x + 1) * chunk_size, :, :, :])
            adv_accum += adv
            r1_accum += r1
        
        
        self.optimizer.step()
        
        return adv_accum + r1_accum, r1_accum
        
    def eval_G_once(self, lr, real):
        sr = self.G(lr, 0)
        fidelity = FidelityLoss(sr, lr)

        output_f = self.dis(sr)
        output_r = self.dis(real)
        
        adv = self.regularization_weight * RelativisticLoss(output_f, output_r) / self.grad_accum
        fidelity = self.fidelity_weight * fidelity / self.grad_accum
        
        (adv + fidelity).backward()
        
        return adv.detach().item(), fidelity.detach().item()
 
    def forward(self, lr, real):
        self.dis.requires_grad = False
        self.G.requires_grad = True
        
        self.G.zero_grad()
        
        batch_size = lr.shape[0]
        chunk_size = batch_size // self.grad_accum
        
        adv_accum = 0
        fidelity_accum = 0
        
        
        for x in range(self.grad_accum):
            adv, fidelity = self.eval_G_once(lr[x * chunk_size : (x + 1) * chunk_size, :, :, :], real[x * chunk_size : (x + 1) * chunk_size, :, :, :])
            adv_accum += adv
            fidelity_accum += fidelity
        
        return adv_accum + fidelity_accum, fidelity_accum




    
    
    
    
    def state_dict(self, *args, **kwargs):
        state_discriminator = self.dis.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()

        return dict(**state_discriminator, **state_optimizer)