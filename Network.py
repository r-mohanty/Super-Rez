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

class BiasedActivation(nn.Module):
    Gain = math.sqrt(2)
    
    def __init__(self, InputUnits, ConvolutionalLayer=True):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
        self.ConvolutionalLayer = ConvolutionalLayer
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if self.ConvolutionalLayer else x + self.Bias.view(1, -1)
        return nn.functional.silu(y)

class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(InputChannels)

      def forward(self, x, ActivationMaps):
          y = self.LinearLayer1(ActivationMaps)
          y = self.NonLinearity1(y)
          
          y = self.LinearLayer2(y)
          y = x + y
          
          return y, self.NonLinearity2(y) 
        
class GeneratorPrologLayer(nn.Module):
    def __init__(self, OutputChannels, FeatureChannels, ReceptiveField):
        super(GeneratorPrologLayer, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Conv2d(3, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.ToFeatures = MSRInitializer(nn.Conv2d(OutputChannels, FeatureChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        
    def forward(self, x):
        x = self.LinearLayer(x)
        ActivationMaps = self.NonLinearity(x)
        
        return x, ActivationMaps, self.ToFeatures(ActivationMaps)

class GeneratorStage(nn.Module):
      def __init__(self, InputChannels, FeatureChannels, Blocks, CompressionFactor, ReceptiveField):
          super(GeneratorStage, self).__init__()
          
          self.MainBlocks = nn.ModuleList([GeneratorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks)])
          self.ToFeatures = MSRInitializer(nn.Conv2d(InputChannels, FeatureChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
          
      def forward(self, x, ActivationMaps):
          for Block in self.MainBlocks:
              x, ActivationMaps = Block(x, ActivationMaps)
        
          return x, ActivationMaps, self.ToFeatures(ActivationMaps)
      
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
      
class GeneratorUpsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorUpsampleBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels * 4, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels * 4)
          self.NonLinearity3 = BiasedActivation(OutputChannels)
          
          self.Resampler = Upsampler()
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

      def forward(self, x, ActivationMaps):
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)
          
          y = self.LinearLayer1(ActivationMaps)
          y = self.LinearLayer2(self.NonLinearity1(y))
          y = self.Resampler(self.NonLinearity2(y))
          
          y = self.LinearLayer3(y)
          y = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + y
          
          return y, self.NonLinearity3(y)

class Generator(nn.Module):
    def __init__(self, StemWidth=256, FeatureWidths=[512, 256, 128], BlocksPerStage=[16, 16, 16, 16], CompressionFactor=4, ReceptiveField=3):
        super(Generator, self).__init__()
        
        self.Stem = GeneratorPrologLayer(StemWidth, FeatureWidths[0], ReceptiveField)
        self.Stages = nn.ModuleList([GeneratorStage(StemWidth, FeatureWidths[0], x, CompressionFactor, ReceptiveField) for x in BlocksPerStage])
        
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
        ActivatedFeatures = self.FeatureNonLinearity(AggregatedFeatures)
        
        for Upsample, Aggregate in zip(self.Upsamplers, self.ToRGB):
            AggregatedFeatures, ActivatedFeatures = Upsample(AggregatedFeatures, ActivatedFeatures)
            ImageOutput = nn.functional.interpolate(ImageOutput, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + Aggregate(ActivatedFeatures)
        
        return ImageOutput










#### quick test ####
# Network2x = Generator(FeatureWidths=[512, 256])
# Network4x = Generator()

# print('params: ' + str(sum(p.numel() for p in Network4x.parameters() if p.requires_grad)))

# x = torch.rand((12, 3, 32, 32))
# y = Network4x(x)
# print(y.shape)
