import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn import init
from sync_batchnorm import SynchronizedBatchNorm2d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).'
              % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc, ks=3):
        super().__init__()

        self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv   = self.mlp_shared(segmap)
        gamma  = self.mlp_gamma(actv)
        beta   = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        semantic_nc = 125

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        
        self.conv_0 = spectral_norm(self.conv_0)
        self.conv_1 = spectral_norm(self.conv_1)
        if self.learned_shortcut:
            self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(fin, semantic_nc)
        self.norm_1 = SPADE(fmiddle, semantic_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, semantic_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x,  seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADEGenerator(BaseNetwork):
    def __init__(self):
        super(SPADEGenerator, self).__init__()
        self.num_up_layers = 7
        nf = 64
        self.nf = nf
        self.z_dim = 256

        self.sw, self.sh = self.compute_latent_vector_size(self.num_up_layers)

        # In case of VAE, we will sample from random z vector
        self.fc = nn.Linear(self.z_dim, 16 * nf * self.sw * self.sh)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf)
        
        final_nc = nf
        
        # MOST MOST MOST MOST MOST MOST MOST MOST MOST MOST MOST
        if self.num_up_layers == 7:
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2)
            final_nc = nf // 2
        


        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self,num_up_layers ):
        sw = 256 // (2**num_up_layers)
        sh = round(sw)
        return sw, sh

    def forward(self, input, z=None):
        seg = input

        # we sample z from unit normal and reshape the tensor
        if z is None:
            # z = torch.randn(input.size(0), self.z_dim,
            #                 dtype=torch.float32, device=input.get_device())
            z = torch.randn(input.size(0), self.z_dim,dtype=torch.float32)
            
        x = self.fc(z)
        x = x.view(-1, 16 * self.nf, self.sh, self.sw)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.num_up_layers > 5:
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)
        
        if self.num_up_layers == 7:
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x



trained_generator = SPADEGenerator()
trained_generator.load_state_dict(torch.load('./gui/generator230.pt', map_location='cpu'))

def segmentation2space(seg_path):
    seg_ = plt.imread(seg_path)
    seg = torch.round(torch.tensor(seg_) * 255)

    # seg = primary_transform(seg)

    N = 5
    seg = torch.floor(torch.floor((seg*(N/256)))*(255/N)).int()
    qnt = ((seg[...,0] + seg[...,1] * N + seg[...,2] * N * N) * N // 255).long()
    sem = torch.nn.functional.one_hot(qnt, N**3).permute(2,0,1)
    sem = sem.float().unsqueeze(0)


    generated_output = trained_generator(
        sem.to(device),
        torch.randn(1, 256,dtype=torch.float32).to(device)
    )

    generated_output = torch.round((generated_output + 1) * 127).int()
    generated_output = generated_output[0].permute(1,2,0).detach().cpu().numpy()

    # fig, axs = plt.subplots(1, 2 ,figsize=(15, 15))
    # axs[0].imshow(generated_output)
    # axs[0].set_title(f'Generated image')
    # axs[1].imshow(seg)
    # axs[1].set_title(f'Segmentation image')
    # plt.show()

    return generated_output