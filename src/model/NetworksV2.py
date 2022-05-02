import math
import numpy
import torch
import torch.nn as nn

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class NoiseInjector(nn.Module):
    Sampler = lambda x: torch.randn(*x.shape, device=x.device)
    
    def __init__(self, InputChannels):
        super(NoiseInjector, self).__init__()
        
        self.Scale = nn.Parameter(torch.empty(InputChannels))
        self.Scale.data.zero_()
        
    def forward(self, x):
        return self.Scale.view(1, -1, 1, 1) * NoiseInjector.Sampler(x) + x
        
class BiasedActivation(nn.Module):
    Gain = math.sqrt(2)
    Function = nn.functional.mish
    
    def __init__(self, InputUnits, ConvolutionalLayer=True):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
        self.ConvolutionalLayer = ConvolutionalLayer
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if self.ConvolutionalLayer else x + self.Bias.view(1, -1)
        return BiasedActivation.Function(y)

class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NoiseLayer1 = NoiseInjector(CompressedChannels)
          self.NoiseLayer2 = NoiseInjector(InputChannels)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(InputChannels)
          
      def forward(self, x, ActivationMaps):
          y = self.LinearLayer1(ActivationMaps)
          y = self.NonLinearity1(self.NoiseLayer1(y))
          
          y = self.LinearLayer2(y)
          y = x + y
          
          return y, self.NonLinearity2(self.NoiseLayer2(y))

class DiscriminatorBlock(nn.Module):
      def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
          super(DiscriminatorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          y = self.LinearLayer2(self.NonLinearity2(y))
          
          return x + y

def CreateLowpassKernel():
    Kernel = numpy.array([[1., 2., 1.]])
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    Kernel = Kernel / torch.sum(Kernel)
    return Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1])

class Upsampler(nn.Module):
      def __init__(self):
          super(Upsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel())
          
      def forward(self, x):
          x = nn.functional.pixel_shuffle(x, 2)
          y = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
          
          return nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)
          
class Downsampler(nn.Module):
      def __init__(self):
          super(Downsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel())
          
      def forward(self, x):
          y = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
          y = nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)

          return nn.functional.pixel_unshuffle(y, 2)

class GeneratorUpsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorUpsampleBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels * 4, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NoiseLayer1 = NoiseInjector(CompressedChannels)
          self.NoiseLayer2 = NoiseInjector(CompressedChannels)
          self.NoiseLayer3 = NoiseInjector(OutputChannels)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          self.NonLinearity3 = BiasedActivation(OutputChannels)
          
          self.Resampler = Upsampler()
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

      def forward(self, x, ActivationMaps):
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)
          
          y = self.LinearLayer1(ActivationMaps)
          y = self.LinearLayer2(self.NonLinearity1(self.NoiseLayer1(y)))
          y = self.NonLinearity2(self.NoiseLayer2(self.Resampler(y)))
          
          y = self.LinearLayer3(y)
          y = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + y
          
          return y, self.NonLinearity3(self.NoiseLayer3(y))

class DiscriminatorDownsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField):
          super(DiscriminatorDownsampleBlock, self).__init__()
          
          CompressedChannels = OutputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels * 4, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          self.NonLinearity3 = BiasedActivation(CompressedChannels)
          
          self.Resampler = Downsampler()
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          
          y = self.Resampler(self.NonLinearity2(y))
          y = self.NonLinearity3(self.LinearLayer2(y))
          y = self.LinearLayer3(y)
          
          x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True, recompute_scale_factor=True)
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)

          return x + y
     
class GeneratorStage(nn.Module):
      def __init__(self, InputChannels, FeatureChannels, Blocks, CompressionFactor, ReceptiveField):
          super(GeneratorStage, self).__init__()
          
          self.MainBlocks = nn.ModuleList([GeneratorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks)])
          self.ToFeatures = MSRInitializer(nn.Conv2d(InputChannels, FeatureChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
          
      def forward(self, x, ActivationMaps):
          for Block in self.MainBlocks:
              x, ActivationMaps = Block(x, ActivationMaps)
        
          return x, ActivationMaps, self.ToFeatures(ActivationMaps)

class DiscriminatorStage(nn.Module):
      def __init__(self, InputChannels, OutputChannels, Blocks, CompressionFactor, ReceptiveField):
          super(DiscriminatorStage, self).__init__()

          self.BlockList = nn.ModuleList([DiscriminatorDownsampleBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField)] + [DiscriminatorBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)])
        
      def forward(self, x):
          for Block in self.BlockList:
              x = Block(x)
          return x

class GeneratorPrologLayer(nn.Module):
    def __init__(self, OutputChannels, FeatureChannels, ReceptiveField):
        super(GeneratorPrologLayer, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Conv2d(3, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.NoiseLayer = NoiseInjector(OutputChannels)
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.ToFeatures = MSRInitializer(nn.Conv2d(OutputChannels, FeatureChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        
    def forward(self, x):
        x = self.LinearLayer(x)
        ActivationMaps = self.NonLinearity(self.NoiseLayer(x))
        
        return x, ActivationMaps, self.ToFeatures(ActivationMaps)

class DiscriminatorEpilogLayer(nn.Module):
      def __init__(self, InputChannels, BasisSize, LatentDimension):
          super(DiscriminatorEpilogLayer, self).__init__()
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=BasisSize, stride=1, padding=0, groups=InputChannels, bias=False))
          self.LinearLayer2 = MSRInitializer(nn.Linear(InputChannels, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x)).view(x.shape[0], -1)
          return self.NonLinearity2(self.LinearLayer2(y))

class Generator(nn.Module):
    def __init__(self, StemWidth=256, FeatureWidths=[512, 256, 128], BlocksPerStage=[16, 16, 16, 16], CompressionFactor=4, ReceptiveField=3):
        super(Generator, self).__init__()
        
        self.Stem = GeneratorPrologLayer(StemWidth, FeatureWidths[0], ReceptiveField)
        self.Stages = nn.ModuleList([GeneratorStage(StemWidth, FeatureWidths[0], x, CompressionFactor, ReceptiveField) for x in BlocksPerStage])
        
        self.FeatureNoiseLayer = NoiseInjector(FeatureWidths[0])
        self.FeatureNonLinearity = BiasedActivation(FeatureWidths[0])

        Upsamplers = []
        ToRGB = []
        for x in range(len(FeatureWidths) - 1):
            Upsamplers += [GeneratorUpsampleBlock(FeatureWidths[x], FeatureWidths[x + 1], CompressionFactor, ReceptiveField)]
            ToRGB += [MSRInitializer(nn.Conv2d(FeatureWidths[x + 1], 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)]
        self.Upsamplers = nn.ModuleList(Upsamplers)
        self.ToRGB = nn.ModuleList(ToRGB)
        
    def forward(self, x):
        ImageOutput = x
        
        x, ActivationMaps, AggregatedFeatures = self.Stem(x)
        for Stage in self.Stages:
            x, ActivationMaps, FeatureResidual  = Stage(x, ActivationMaps)
            AggregatedFeatures += FeatureResidual
        ActivatedFeatures = self.FeatureNonLinearity(self.FeatureNoiseLayer(AggregatedFeatures))
        
        for Upsample, Aggregate in zip(self.Upsamplers, self.ToRGB):
            AggregatedFeatures, ActivatedFeatures = Upsample(AggregatedFeatures, ActivatedFeatures)
            ImageOutput = nn.functional.interpolate(ImageOutput, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + Aggregate(ActivatedFeatures)
        
        return ImageOutput

class Discriminator(nn.Module):
    def __init__(self, BasisSize=4, LatentDimension=512, EpilogWidth=1024, StageWidths=[128, 256, 256, 256, 512, 512, 512, 1024], BlocksPerStage=[2, 2, 2, 2, 2, 2, 2, 2], CompressionFactor=4, ReceptiveField=3):
        super(Discriminator, self).__init__()
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, StageWidths[0], kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        
        MainLayers = []
        ExtendedStageWidths = StageWidths + [EpilogWidth]
        for x in range(len(StageWidths)):
            MainLayers += [DiscriminatorStage(ExtendedStageWidths[x], ExtendedStageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField)]
        self.MainLayers = nn.ModuleList(MainLayers)
        
        self.EpilogLayer = DiscriminatorEpilogLayer(EpilogWidth, BasisSize, LatentDimension)
        self.CriticLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        
    def forward(self, x):
        x = self.FromRGB(x)

        for Layer in self.MainLayers:
            x = Layer(x)
        
        x = self.EpilogLayer(x)
        return self.CriticLayer(x).view(x.shape[0])









#### quick test ####
# Network2x = Generator(FeatureWidths=[512, 256])
# Network4x = Generator()

# D = Discriminator(BasisSize=3, StageWidths=[256, 256, 512, 512, 512, 1024], BlocksPerStage=[1, 1, 2, 2, 2, 1])

# print('G params: ' + str(sum(p.numel() for p in Network4x.parameters() if p.requires_grad)))
# print('D params: ' + str(sum(p.numel() for p in D.parameters() if p.requires_grad)))

# x = torch.rand((4, 3, 48, 48))
# y = Network4x(x)
# c = D(y)
# print(y.shape)
# print(c.shape)