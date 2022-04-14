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

class DiscriminatorBlock(nn.Module):
      def __init__(self, InputChannels, ReceptiveField=3):
          super(DiscriminatorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          y = self.LinearLayer2(self.NonLinearity2(y))
          
          return x + y

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

class Downsampler(nn.Module):
      def __init__(self, InputChannels):
          super(Downsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(InputChannels))
          self.InputChannels = InputChannels
          
      def forward(self, x):
          x = nn.functional.conv2d(nn.functional.pad(x, (2, 1, 2, 1), mode='reflect'), self.Kernel, stride=1, groups=self.InputChannels)
          return nn.functional.pixel_unshuffle(x, 2)

class GeneratorUpsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(GeneratorUpsampleBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels * 4, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(OutputChannels)
          
          self.Resampler = Upsampler(CompressedChannels)
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

      def forward(self, x, ActivationMaps):
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)
          
          y = self.LinearLayer1(ActivationMaps)
          y = self.NonLinearity1(self.Resampler(y))
          
          y = self.LinearLayer2(y)
          y = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) + y
          
          return y, self.NonLinearity2(y)

class DiscriminatorDownsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, ReceptiveField=3):
          super(DiscriminatorDownsampleBlock, self).__init__()
          
          CompressedChannels = OutputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels * 4, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          
          self.Resampler = Downsampler(CompressedChannels)
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          
          y = self.Resampler(self.NonLinearity2(y))
          y = self.LinearLayer2(y)
          
          x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)

          return x + y
     
class GeneratorStage(nn.Module):
      def __init__(self, InputChannels, OutputChannels, Blocks, ReceptiveField=3):
          super(GeneratorStage, self).__init__()
          
          self.BlockList = nn.ModuleList([GeneratorBlock(InputChannels, ReceptiveField) for _ in range(Blocks)] + [GeneratorUpsampleBlock(InputChannels, OutputChannels, ReceptiveField)])
          
      def forward(self, x, ActivationMaps):
          for Block in self.BlockList:
              x, ActivationMaps = Block(x, ActivationMaps)
          return x, ActivationMaps
          
class DiscriminatorStage(nn.Module):
      def __init__(self, InputChannels, OutputChannels, Blocks, ReceptiveField=3):
          super(DiscriminatorStage, self).__init__()

          self.BlockList = nn.ModuleList([DiscriminatorDownsampleBlock(InputChannels, OutputChannels, ReceptiveField)] + [DiscriminatorBlock(OutputChannels, ReceptiveField) for _ in range(Blocks)])
        
      def forward(self, x):
          for Block in self.BlockList:
              x = Block(x)
          return x
        
class GeneratorOpeningLayer(nn.Module):
    def __init__(self, LatentDimension, OutputChannels):
        super(GeneratorOpeningLayer, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty((OutputChannels, 4, 4)))
        self.LinearLayer = MSRInitializer(nn.Linear(LatentDimension, OutputChannels, bias=False))
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.Basis.data.normal_(0, SiLUGain)
        
    def forward(self, w):
        x = self.LinearLayer(w).view(w.shape[0], -1, 1, 1)
        y = self.Basis.view(1, -1, 4, 4) * x
        return y, self.NonLinearity(y)
     
class DiscriminatorClosingLayer(nn.Module):
      def __init__(self, InputChannels, LatentDimension):
          super(DiscriminatorClosingLayer, self).__init__()
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
          self.LinearLayer2 = MSRInitializer(nn.Linear(InputChannels, LatentDimension, bias=False), ActivationGain=SiLUGain)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x)).view(x.shape[0], -1)
          return self.NonLinearity2(self.LinearLayer2(y))

class FullyConnectedBlock(nn.Module):
    def __init__(self, LatentDimension):
        super(FullyConnectedBlock, self).__init__()

        self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=SiLUGain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        
        return x + y
           
class MappingBlock(nn.Module):
      def __init__(self, LatentDimension):
          super(MappingBlock, self).__init__()
          
          self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=SiLUGain)
          
          self.Layer2To3 = FullyConnectedBlock(LatentDimension)
          self.Layer4To5 = FullyConnectedBlock(LatentDimension)
          self.Layer6To7 = FullyConnectedBlock(LatentDimension)
          
          self.NonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          self.LinearLayer8 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=SiLUGain)
          
          self.ClosingNonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, z):
          w = self.LinearLayer1(z)
          
          w = self.Layer2To3(w)
          w = self.Layer4To5(w)
          w = self.Layer6To7(w)
          
          w = self.LinearLayer8(self.NonLinearity(w))
          
          return self.ClosingNonLinearity(w)
      
def ToRGB(InputChannels, ResidualComponent=False):
    return MSRInitializer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0 if ResidualComponent else 1)

class Generator(nn.Module):
    def __init__(self, LatentDimension):
        super(Generator, self).__init__()
        
        self.LatentLayer = MappingBlock(LatentDimension)
        
        self.Layer4x4 = GeneratorOpeningLayer(LatentDimension, 512 * 2)
        self.ToRGB4x4 = ToRGB(512 * 2)
        
        self.Layer8x8 = GeneratorStage(512 * 2, 512 * 2, 1)
        self.ToRGB8x8 = ToRGB(512 * 2, ResidualComponent=True)
        
        self.Layer16x16 = GeneratorStage(512 * 2, 512, 1)
        self.ToRGB16x16 = ToRGB(512, ResidualComponent=True)
        
        self.Layer32x32 = GeneratorStage(512, 512, 1)
        self.ToRGB32x32 = ToRGB(512, ResidualComponent=True)
        
        self.Layer64x64 = GeneratorStage(512, 512, 1)
        self.ToRGB64x64 = ToRGB(512, ResidualComponent=True)
        
        self.Layer128x128 = GeneratorStage(512, 256, 1)
        self.ToRGB128x128 = ToRGB(256, ResidualComponent=True)
        
    def forward(self, z, EnableLatentMapping=True):
        w = self.LatentLayer(z) if EnableLatentMapping else z
        
        y, ActivationMaps = self.Layer4x4(w)
        Output4x4 = self.ToRGB4x4(ActivationMaps)

        y, ActivationMaps = self.Layer8x8(y, ActivationMaps)
        Output8x8 = nn.functional.interpolate(Output4x4, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB8x8(ActivationMaps)

        y, ActivationMaps = self.Layer16x16(y, ActivationMaps)
        Output16x16 = nn.functional.interpolate(Output8x8, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB16x16(ActivationMaps)

        y, ActivationMaps = self.Layer32x32(y, ActivationMaps)
        Output32x32 = nn.functional.interpolate(Output16x16, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB32x32(ActivationMaps)

        y, ActivationMaps = self.Layer64x64(y, ActivationMaps)
        Output64x64 = nn.functional.interpolate(Output32x32, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB64x64(ActivationMaps)

        y, ActivationMaps = self.Layer128x128(y, ActivationMaps)
        Output128x128 = nn.functional.interpolate(Output64x64, scale_factor=2, mode='bilinear', align_corners=False) + self.ToRGB128x128(ActivationMaps)
        
        return Output128x128

class Discriminator(nn.Module):
    def __init__(self, LatentDimension):
        super(Discriminator, self).__init__()
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False), ActivationGain=SiLUGain)
        
        self.Layer128x128 = DiscriminatorStage(256, 512, 1)
        self.Layer64x64 = DiscriminatorStage(512, 512, 1)
        self.Layer32x32 = DiscriminatorStage(512, 512, 1)
        self.Layer16x16 = DiscriminatorStage(512, 512 * 2, 1)
        self.Layer8x8 = DiscriminatorStage(512 * 2, 512 * 2, 1)
        self.Layer4x4 = DiscriminatorClosingLayer(512 * 2, LatentDimension)
        
        self.CriticLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        
    def forward(self, x):
        x = self.Layer128x128(self.FromRGB(x))
        x = self.Layer64x64(x)
        x = self.Layer32x32(x)
        x = self.Layer16x16(x)
        x = self.Layer8x8(x)
        x = self.Layer4x4(x)
        
        return self.CriticLayer(x).squeeze()