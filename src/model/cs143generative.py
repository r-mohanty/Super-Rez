from model import common
from model.NetworksV2 import Generator

import torch.nn as nn

def make_model(args, parent=False):
    return CS143Generator(args)

class CS143Generator(nn.Module):
    def __init__(self, args):
        super(CS143Generator, self).__init__()
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        
        if args.scale[0] == 2:
            self.Main = Generator(FeatureWidths=[512, 256])
        elif args.scale[0] == 4:
            self.Main = Generator()
            
    def forward(self, x):
        x = self.sub_mean(x)
        
        x = self.Main(x)
        
        x = self.add_mean(x)

        return x 




