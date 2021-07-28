from layers import *
import torch
from torch import nn
from rocket.ignite.layers import downsample3D, upsample3D,downsample2D,upsample2D, LateralConnect2D, LateralConnect3D
from rocket.images.preprocess import stack_frames
import torchviz
from torchsummary  import summary
from torchvision.transforms import Resize,InterpolationMode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
        def __init__(self):
                super(Encoder, self).__init__()
                self.downs = [
                        downsample2D(3, 32, [4, 4], [ 2, 2], padding=[1,1], apply_batchnorm=False),  # (bs, 4, 128, 128, 64)
                        nn.MaxPool2d((2)), 
                        downsample2D(32, 64, [ 4, 4], [ 2, 2], padding=[1,1]),  # (bs, 4, 64, 64, 128)
                        nn.MaxPool2d((2)),
                        downsample2D(64, 128, [ 4, 4], [ 2, 2], padding=[1,1]),  # (bs, 4, 4, 32, 256)
                        nn.MaxPool2d((2)),
                        downsample2D(128, 256, [ 4, 4], [ 2, 2], padding=[1,1]),  # (bs, 4, 4, 32, 256)
                        nn.MaxPool2d((2),padding=1),
                        downsample2D(256, 512, [ 4, 4], [ 2, 2], padding=[1,1],apply_batchnorm=False),  # (bs, 4, 4, 32, 256)
                        # downsample2D(512, 512, [ 4, 4], [ 1, 1], padding=[0,0]),  # (bs, 4, 4, 32, 256)

                ]
                self.model = nn.Sequential(*self.downs,
                                        nn.Flatten())

        def forward(self, x) :
                x =self.model(x)
                return x


class Decoder(nn.Module):
    def __init__(self, input_size,feature_size,channel_size,output_size=(640,480)):
        super(Decoder, self).__init__()
        self.ups = [
            upsample2D(input_size,feature_size*8,1,2,1,dilation=4,activation=nn.ReLU(True),bias=False),
            upsample2D(feature_size*8,feature_size*8,2,2,1,activation=nn.ReLU(True),bias=False),

            upsample2D(feature_size*8,feature_size*4,4,2,1,activation=nn.ReLU(True),bias=False),
            upsample2D(feature_size*4,feature_size*4,2,2,1,activation=nn.ReLU(True),bias=False),

            upsample2D(feature_size*4,feature_size*2,4,2,1,activation=nn.ReLU(True),bias=False),
            upsample2D(feature_size*2,feature_size*2,2,2,1,activation=nn.ReLU(True),bias=False),

            upsample2D(feature_size*2,feature_size,4,2,1,activation=nn.ReLU(True),bias=False),
            upsample2D(feature_size,feature_size,2,2,1,activation=nn.ReLU(True),bias=False),
            upsample2D(feature_size,channel_size,2,2,1,activation=nn.ReLU(True),bias=False)



        ]
        self.output_size = output_size
        self.model = nn.Sequential(*self.ups)
        self.last = nn.ConvTranspose2d(channel_size, channel_size, 4, stride=0, padding=1,bias=False)
        self.activ = nn.ReLU(True)

        # self.last = upsample2D(channel_size,channel_size,4,0,1,activation=nn.ReLU(True),bias=False)
    
    def forward(self,x):
        x = self.model(x)
        x = Resize(self.output_size,interpolation=InterpolationMode.NEAREST)(x)
        return x