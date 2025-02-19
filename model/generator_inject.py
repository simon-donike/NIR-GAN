import torch
import torch.nn as nn
import functools
from model.networks import ResnetBlock


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
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
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
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
        embeds = embeds.repeat(1, 128, 1, 1)        
        
        # Combine feature map with context
        x = x + embeds
        
        # Apply remaining layers
        x = self.model[6:](x)  
        return x
    
    
    
if __name__=="__main__":
        
    # get Model    
    m = ResnetGenerator(3, 1)
    
    # print FC layer size
    fc_params = sum(p.numel() for p in m.fc.parameters() if p.requires_grad)
    print(f"Parameters in fc layer: {fc_params}")
    
    # print whole Model layer size
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # test forward step
    embeds = torch.rand(2,256)
    a = torch.rand(2,3,512,512)
    
    #pred = m(a)
    pred_inj = m(a,embeds)
    
    

