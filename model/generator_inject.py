import torch
import torch.nn as nn
import functools
from model.networks import ResnetBlock


class ResnetGenerator_inject(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    
    # Addition: Inject embeddings into the model after downsampling layers
    """
    """
    def __init__(self, input_nc, output_nc,inject_style="multiply",post_correction=False,
                 ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6, padding_type='reflect'):
    """
    """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """
    def __init__(self,config,norm_layer,n_blocks=9):
        
        padding_type='reflect'
        
        # extract info from config
        input_nc = config.base_configs.input_nc
        output_nc = config.base_configs.output_nc
        ngf = config.base_configs.ngf
        use_dropout = not config.base_configs.no_dropout
        inject_style = config.satclip.satclip_inject_style
        post_correction = config.satclip.post_correction
        
        
        self.inject_style=inject_style
        assert(n_blocks >= 0)
        super(ResnetGenerator_inject, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        # define injection of embeddings        
        self.embed_fc_ou_square = 128
        self.fc = nn.Linear(in_features=256, out_features=self.embed_fc_ou_square*self.embed_fc_ou_square)  # Example size, adjust accordingly
        
        # define learned scaling parameter for embeddings
        self.scale_param = nn.Parameter(torch.tensor(0.1))
        
        self.post_correction = post_correction
        if self.post_correction:
            print("Setting Post-Correction Parameter.")
            self.post_correction_param = nn.Parameter(torch.tensor(1.))

        # build model
        self.model = nn.Sequential(*model)
    
    def forward(self, input, embeds):
        # Apply initial layers up to the desired injection point
        x = self.model[:6](input)
        
        # get embedds of shape B,65536
        embeds = self.fc(embeds)
        
        # Reshape to B,1,w,h from 1D tensor
        embeds = embeds.view(-1, 1, self.embed_fc_ou_square, self.embed_fc_ou_square)  # Reshape to B,1,256,256
        
        # Upsample to match feature map size
        embeds = nn.functional.interpolate(embeds, size=(x.shape[-1], x.shape[-2]), mode='bilinear', align_corners=False)
        
        # Repeat channel dimension to match feature map
        embeds = embeds.repeat(1, x.shape[-3], 1, 1)        
                
        # Combine feature map with context
        if self.inject_style == "add":
            x = x + (self.scale_param * embeds)
        elif self.inject_style == "multiply":
            x = x * (1 + self.scale_param * embeds)
        
        # Apply remaining layers
        x = self.model[6:](x)  
        
        # apply post-correction via learnable parameter
        if self.post_correction:
            x = x * self.post_correction_param
        return x
    

from model.networks import get_norm_layer, init_net
"""
def define_G_inject(input_nc, output_nc,inject_style,
                    post_correction=False,
                    ngf=64, netG="resnet_9blocks", norm='batch', use_dropout=False,
                    init_type='normal', init_gain=0.02, gpu_ids=[]):
"""
def define_G_inject(config):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        inject_style (str) -- the style of injection: add | multiply
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    # extract info from config
    input_nc = config.base_configs.input_nc
    output_nc = config.base_configs.output_nc
    ngf = config.base_configs.ngf
    netG = config.base_configs.netG
    norm = config.base_configs.norm
    use_dropout = not config.base_configs.no_dropout
    init_type = config.base_configs.init_type
    init_gain = config.base_configs.init_gain
    gpu_ids=[]
    inject_style = config.satclip.satclip_inject_style
    post_correction = config.satclip.post_correction
    
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        n_blocks = 9 # hardcoded bc thats the string definition
        """
        net = ResnetGenerator_inject(input_nc, output_nc,
                                     inject_style, post_correction,ngf, 
                                     norm_layer=norm_layer, use_dropout=use_dropout,
                                     n_blocks=9)
        """
        net = ResnetGenerator_inject(config,norm_layer=norm_layer,n_blocks=n_blocks)
    else:
        net = None
        raise NotImplementedError('Generator model name [%s] is not recognized. Only resnet_9blocks for SatCLIP.' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)

    
if __name__=="__main__":
        
    # get Model    
    m = ResnetGenerator_inject(3, 1,"multiply",post_correction=True)
    
    # print FC layer size
    fc_params = sum(p.numel() for p in m.fc.parameters() if p.requires_grad)
    print(f"Parameters in fc layer: {fc_params}")
    
    # print whole Model layer size
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # test forward step
    embeds = torch.rand(2,256)
    a = torch.rand(2,3,512,512)
        
    pred_inj = m(a,embeds)
    
    

