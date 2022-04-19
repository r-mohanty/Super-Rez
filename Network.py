import math
import numpy
import torch
import torch.nn as nn

CompressionFactor = 4
SiLUGain = math.sqrt(2)

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class BiasedActivation(nn.Module):
    def __init__(self, InputUnits, ConvolutionalLayer=True):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
        self.ConvolutionalLayer = ConvolutionalLayer
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if self.ConvolutionalLayer else x + self.Bias.view(1, -1)
        return nn.functional.silu(y)
    
class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, ReceptiveField=3):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(InputChannels)

      def forward(self, x, ActivationMaps):
          y = self.LinearLayer1(ActivationMaps)
          y = self.NonLinearity1(y)
          
          y = self.LinearLayer2(y)
          y = x + y
          
          return y, self.NonLinearity2(y)
      
def CreateLowpassKernel(InputChannels):
    Kernel = numpy.array([[1., 3., 3., 1.]])
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    Kernel = Kernel / torch.sum(Kernel)
    return Kernel.repeat((InputChannels, 1, 1, 1))

class Upsampler(nn.Module):
      def __init__(self, InputChannels):
          super(Upsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(InputChannels))
          self.InputChannels = InputChannels
          
      def forward(self, x):
          x = nn.functional.pixel_shuffle(x, 2)
          return nn.functional.conv2d(nn.functional.pad(x, (2, 1, 2, 1), mode='reflect'), self.Kernel, stride=1, groups=self.InputChannels)

class GeneratorAggregationBlock(nn.Module):
      def __init__(self, InputChannels, ReceptiveField=3):
          super(GeneratorAggregationBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels * 4, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=SiLUGain)
          self.NonLinearity = BiasedActivation(CompressedChannels * 4)

          self.ToRGB = MSRInitializer(nn.Conv2d(CompressedChannels, 3, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          self.Resampler = Upsampler(CompressedChannels)
          
      def forward(self, ActivationMaps):
          y = self.LinearLayer(ActivationMaps)
          y = self.Resampler(self.NonLinearity(y))

          return self.ToRGB(y)

class GeneratorStage(nn.Module):
      def __init__(self, InputChannels, Blocks, ReceptiveField=3):
          super(GeneratorStage, self).__init__()
          
          self.MainBlocks = nn.ModuleList([GeneratorBlock(InputChannels, ReceptiveField) for _ in range(Blocks)])
          self.Aggregate = GeneratorAggregationBlock(InputChannels, ReceptiveField)
          
      def forward(self, x, ActivationMaps):
          for Block in self.MainBlocks:
              x, ActivationMaps = Block(x, ActivationMaps)
        
          return x, ActivationMaps, self.Aggregate(ActivationMaps)
      
class GeneratorOpeningLayer(nn.Module):
    def __init__(self, OutputChannels, ReceptiveField=3):
        super(GeneratorOpeningLayer, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Conv2d(3, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.Aggregate = GeneratorAggregationBlock(OutputChannels, ReceptiveField)
        
    def forward(self, x):
        x = self.LinearLayer(x)
        ActivationMaps = self.NonLinearity(x)
        
        return x, ActivationMaps, self.Aggregate(ActivationMaps)

class Generator(nn.Module):
    def __init__(self, StemWidth=256, BlocksPerStage=[16, 16, 16, 16]):
        super(Generator, self).__init__()
        
        self.Stem = GeneratorOpeningLayer(StemWidth)
        self.Stages = nn.ModuleList([GeneratorStage(StemWidth, x) for x in BlocksPerStage])
        
    def forward(self, x):
        ImageOutput = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x, ActivationMaps, HighFrequencyResidual = self.Stem(x)
        ImageOutput += HighFrequencyResidual
        
        for Stage in self.Stages:
            x, ActivationMaps, HighFrequencyResidual = Stage(x, ActivationMaps)
            ImageOutput += HighFrequencyResidual
            
        return ImageOutput










#### quick test ####
# m = Generator()

# print('params: ' + str(sum(p.numel() for p in m.parameters() if p.requires_grad)))

# x = torch.rand((12, 3, 32, 32))
# y = m(x)
# print(y.shape)
