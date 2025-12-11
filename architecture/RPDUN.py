
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
import numbers
from torch import einsum


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class SPA_ATSA(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, window_size[0] * window_size[1], window_size[0] * window_size[1]))

        trunc_normal_(self.pos_emb)

        self.Threshold = nn.Conv2d(dim, self.num_heads, kernel_size=3, stride=1,padding=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.w = nn.Parameter(
            torch.Tensor([0.5]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                          b0=self.window_size[0], b1=self.window_size[1]), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        q_ = rearrange(q,'b h (m n) d -> b (h d) m n',m=self.window_size[0],n=self.window_size[1])
        threshold = self.Threshold(q_)
        threshold = self.relu(threshold)
        threshold = rearrange(threshold,'b c m n -> b c (m n)')

        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim_threshold = torch.where(sim < threshold[:,:,:,None],torch.zeros_like(sim), sim)
        sim_threshold = sim_threshold + self.pos_emb
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)*self.w + sim_threshold*(1-self.w)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size[0],
                        w=w // self.window_size[1],
                        b0=self.window_size[0])
        out = self.project_out(out)

        return out


class SPE_ATSA(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_num
                 ):
        super().__init__()
        self.dim = dim
        self.window_num = window_num
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.window_size = [8,8]
        self.pos_emb = nn.Parameter(
           torch.Tensor(1, 1, self.dim, self.dim))
        trunc_normal_(self.pos_emb)

        self.Threshold = nn.Sequential(nn.Conv2d(dim,dim, kernel_size=3, stride=1,padding=1),nn.AdaptiveAvgPool2d((1,1)))
        self.relu = nn.LeakyReLU()
        self.w = nn.Parameter(
            torch.Tensor([0.5]))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        x = x.roll(shifts=4, dims=2).roll(shifts=4, dims=3)

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        h1, h2 = h // self.window_size[0], w // self.window_size[1]
        q, k, v = map(lambda t: rearrange(t, 'b c (h1 h) (h2 w) ->b (h1 h2) c (h w)', h1=h1, h2=h2), (q, k, v))

        q_ = rearrange(q,'b (h1 h2) c (h w) -> (b h1 h2) c h w',h=self.window_size[0],h1=h1)
        threshold = self.Threshold(q_)
        threshold = self.relu(threshold)
        threshold = rearrange(threshold,'(b h1 h2) c h w -> b (h1 h2) c (h w)',h1=h1,h2=h2)
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim_threshold = torch.where(sim < threshold, torch.zeros_like(sim), sim)
        sim_threshold = sim_threshold + self.pos_emb
        sim = sim + self.pos_emb

        attn = sim.softmax(dim=-1) * self.w + sim_threshold * (1 - self.w)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b (h1 h2) c (h w) -> b c (h1 h) (h2 w)', h1=h1, h=h // h1)

        out = self.project_out(out)
        out = out.roll(shifts=-4, dims=2).roll(shifts=-4, dims=3)


        return out

def FFN_FN(ffn_name, dim):
    if ffn_name == "Gated_Dconv_FeedForward1":
        return Gated_Dconv_FeedForward(dim, ffn_expansion_factor=2.66)
    if ffn_name == "Gated_Dconv_FeedForward2":
        return Gated_Dconv_FeedForward(dim, ffn_expansion_factor=2.66)
    elif ffn_name == "FeedForward":
        return FeedForward(dim=dim)


class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)


class ATSBlock(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            window_size=(8, 8),
            window_num=(8, 8),
            num_blocks=2,
            layernorm_type="WithBias",
    ):
        super().__init__()

        self.window_size = window_size
        self.window_num = window_num

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, SPA_ATSA(dim=dim, window_size=window_size, num_heads=num_heads),
                        layernorm_type=layernorm_type),
                PreNorm(dim, SPE_ATSA(dim=dim, num_heads=num_heads, window_num=window_num),
                        layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward1", dim=dim), layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(ffn_name="Gated_Dconv_FeedForward2", dim=dim), layernorm_type=layernorm_type)
            ]))

    def forward(self, x):
        for (SPA, SPE, ffn1,ffn2) in self.blocks:
            x = x + ffn1(x)
            x = x + SPA(x)
            x = x + ffn2(x)
            x = x + SPE(x)

        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, stride=2, kernel_size=2, padding=0, output_padding=0)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ATST(nn.Module):

    def __init__(self,
                 in_dim=28,
                 out_dim=28,
                 dim=28,
                 window_size=(16, 16),
                 window_num=(8, 8),
                 layernorm_type="WithBias",
                 num_blocks=(1, 1, 1, 1, 1)):
        super().__init__()

        self.dim = dim
        self.scales = len(num_blocks)

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.Encoder = nn.ModuleList([
            ATSBlock(
                dim=dim * 2 ** 0,
                num_heads=2 ** 0,
                window_size=window_size,
                window_num=window_num,
                layernorm_type=layernorm_type,
                num_blocks=num_blocks[0],
            ),
            ATSBlock(
                dim=dim * 2 ** 1,
                num_heads=2 ** 1,
                window_size=window_size,
                window_num=window_num,
                layernorm_type=layernorm_type,
                num_blocks=num_blocks[1],
            ),
        ])

        self.BottleNeck = ATSBlock(
            dim=dim * 2 ** 2,
            num_heads=2 ** 2,
            window_size=window_size,
            window_num=window_num,
            layernorm_type=layernorm_type,
            num_blocks=num_blocks[2],
        )

        self.Decoder = nn.ModuleList([
            ATSBlock(
                dim=dim * 2 ** 1,
                num_heads=2 ** 1,
                window_size=window_size,
                window_num=window_num,
                layernorm_type=layernorm_type,
                num_blocks=num_blocks[3],
            ),
            ATSBlock(
                dim=dim * 2 ** 0,
                num_heads=2 ** 0,
                window_size=window_size,
                window_num=window_num,
                layernorm_type=layernorm_type,
                num_blocks=num_blocks[4],
            )
        ])

        self.Downs = nn.ModuleList([
            DownSample(dim * 2 ** 0),
            DownSample(dim * 2 ** 1)
        ])

        self.Ups = nn.ModuleList([
            UpSample(dim * 2 ** 2),
            UpSample(dim * 2 ** 1)
        ])

        self.fusions = nn.ModuleList([
            nn.Conv2d(
                in_channels=dim * 2 ** 2,
                out_channels=dim * 2 ** 1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.Conv2d(
                in_channels=dim * 2 ** 1,
                out_channels=dim * 2 ** 0,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            )
        ])

        self.mapping = nn.Conv2d(dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, c, h_inp, w_inp = x.shape
        hb, wb = 32, 32
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        x1 = self.embedding(x)
        res1 = self.Encoder[0](x1)

        x2 = self.Downs[0](res1)
        res2 = self.Encoder[1](x2)

        x4 = self.Downs[1](res2)
        res4 = self.BottleNeck(x4)

        dec_res2 = self.Ups[0](res4)  # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = torch.cat([dec_res2, res2], dim=1)  # dim * 2 ** 2
        dec_res2 = self.fusions[0](dec_res2)  # dim * 2 ** 2 -> dim * 2 ** 1
        dec_res2 = self.Decoder[0](dec_res2)

        dec_res1 = self.Ups[1](dec_res2)  # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = torch.cat([dec_res1, res1], dim=1)  # dim * 2 ** 1
        dec_res1 = self.fusions[1](dec_res1)  # dim * 2 ** 1 -> dim * 2 ** 0
        dec_res1 = self.Decoder[1](dec_res1)

        out = self.mapping(dec_res1) + x[:, :28, :, :]

        return out[:, :, :h_inp, :w_inp]

def A(x, Phi):
    temp = x * Phi
    y = torch.sum(temp, 1)
    return y


def At(y, Phi):
    temp = torch.unsqueeze(y, 1).repeat(1, Phi.shape[1], 1, 1)
    x = temp * Phi
    return x


def shift_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=step * i, dims=2)
    return inputs


def shift_back_3d(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:, i, :, :] = torch.roll(inputs[:, i, :, :], shifts=(-1) * step * i, dims=2)
    return inputs


def PWDWPWConv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, 1, 1, 0, bias=True),
        nn.GELU(),
        nn.Conv2d(64, 64, 3, 1, 1, bias=True, groups=64),
        nn.GELU(),
        nn.Conv2d(64, out_channels, 1, 1, 0, bias=False)
    )


class HyPaNet(nn.Module):
    def __init__(self):
        super(HyPaNet, self).__init__()
        self.DL = nn.Sequential(
            PWDWPWConv(28 * 2, 28 * 2),
            PWDWPWConv(28 * 2, 28),
        )
        self.down_sample = nn.Conv2d(28, 28 * 2, 3, 2, 1, bias=True)  # (B, 64, H, W) -> (B, 64, H//2, W//2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(28 * 2, 28 * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(28 * 2, 28 * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(28 * 2, 6, 1, padding=0, bias=True),
            nn.Softplus())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, y, phi):
        inp = torch.cat([phi, y], dim=1)
        phi_r = self.DL(inp)  # [b,28,256,310]

        phi = phi + phi_r

        x = self.down_sample(self.relu(phi_r))  # [b,56,128,155]
        x = self.avg_pool(x)  # [b,56,1,1]
        x = self.mlp(x) + 1e-6  # [b,2,1,1]
        mu = x[:, 0, :, :]  # [b,1,1]
        noise_level = x[:, 1:2, :, :]  # [b,1,1,1]
        beta = x[:, 2:3, :, :]
        gamma = x[:, 3:4, :, :]
        lp = x[:, 4:5, :, :]
        rp = x[:, 5:6, :, :]
        return phi, mu, noise_level, beta, gamma, lp, rp


class P(nn.Module):
    """
        to solve min(P) = ||I-PQ||^2 + γ||P-R||^2
        this is a least square problem
        how to solve?
        P* = (gamma*R + I*Q) / (Q*Q + gamma)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, Q, R, beta):
        return ((x * Q + beta * R) / (beta + Q * Q))


class Q(nn.Module):
    """
        to solve min(Q) = ||I-PQ||^2 + λ||Q-L||^2
        Q* = (lamda*L + I*P) / (P*P + lamda)
    """

    def __init__(self):
        super().__init__()
    def forward(self, x, P, L, gamma, Q, i):
        return ((torch.mean(x,dim=1).unsqueeze(1)*torch.mean(P,dim=1).unsqueeze(1)+gamma*L)/(gamma + torch.mean(P,dim=1).unsqueeze(1)*torch.mean(P,dim=1).unsqueeze(1)))



class Resblock(nn.Module):
    def __init__(self, ch):
        super(Resblock, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class Retinex_l(nn.Module):
    def __init__(self):
        super(Retinex_l, self).__init__()
        self.block2 = Resblock(28)
        self.head1 = nn.Conv2d(28+1+1, 28, 3, 1, 1)
        self.head2 = nn.Conv2d(28, 1, 3, 1, 1)

    def forward(self, L):
        L1 = self.head1(L)
        L1 = self.block2(L1)
        L1 = self.head2(L1) + L[:, :1, :, :]
        return L1

class Retinex_d(nn.Module):
    def __init__(self):
        super(Retinex_d, self).__init__()
        self.block1 = Resblock(28)
        self.block2 = Resblock(28)
        self.head1 = nn.Conv2d(28, 28, 3, 1, 1)
        self.head2 = nn.Conv2d(28, 1, 3, 1, 1)

    def forward(self, x):
        R = self.block1(x)
        L = self.block2(x)
        L = self.head2(L)
        R = self.head1(R)
        return L, R


class CMB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.to_a = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Conv2d(dim, dim, 7, 1, 3, groups=dim, bias=False),
        )
        self.to_v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def cal_attention(self, x):
        a, v = self.to_a(x), self.to_v(x)
        out = self.to_out(a * v)
        return out

    def forward(self, x):
        out = self.cal_attention(x)
        return out


class SAB(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.Estimator = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1, bias=False),
            GELU(),
        )
        self.SW = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            GELU(),
        )
        self.out = nn.Conv2d(dim, dim, 1, 1, 0)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight.data, mean=0.0, std=.02)

    def forward(self, f):
        f = self.conv(f)
        out = self.SW(f) * self.Estimator(f).repeat(1, self.dim, 1, 1)
        out = self.out(out)
        return out


class Refine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CMB = PreNorm(dim, CMB(dim=dim))
        self.SAB = PreNorm(dim, SAB(dim=dim))
        self.proj = nn.Conv2d(28, 28, 1, 1, 0)
        self.emb = nn.Conv2d(29, 28, 1, 1, 0)

    def forward(self, x0):
        x = self.emb(x0)
        x = self.CMB(x) + x
        x = self.SAB(x) + x
        x = self.proj(x) + x0[:, :28, :, :]
        return x


class RPDUN(nn.Module):

    def __init__(self, num_iterations=1):
        super(RPDUN, self).__init__()
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.DP = HyPaNet()
        self.num_iterations = num_iterations
        self.denoiser = ATST(in_dim=29, out_dim=28, dim=28, num_blocks=(1, 1, 1, 1, 1))
        self.refine = Refine(28)
        self.Retinexl = Retinex_l()
        self.Retinex_d = Retinex_d()
        self.P = P()
        self.Q = Q()
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initial(self, y, Phi):
        nC, step = 28, 2
        y = y / nC * 2
        bs, row, col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :,
                                                                          step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        return z

    def forward(self, y, input_mask=None):
        phi, Phi_s = input_mask
        z = self.initial(y, phi)  # [b,28,256,310]
        L_list = []
        for i in range(self.num_iterations):
            Phi, mu, noise_level, betat, gamma, lp, rp = self.DP(z, phi)  # mu:[b,1,1]  noise_level:[b,1,1,1]
            Phi_z = A(z, Phi)
            Phi_s = torch.sum(Phi ** 2, 1)
            Phi_s[Phi_s == 0] = 1
            x = z + At(torch.div(y - Phi_z, mu + Phi_s), Phi)
            x = shift_back_3d(x)
            beta_repeat = noise_level.repeat(1, 1, x.shape[2], x.shape[3])
            betat_repeat = betat.repeat(1, 1, x.shape[2], x.shape[3])
            gamma_repeat = gamma.repeat(1, 1, x.shape[2], x.shape[3])
            lp_repeat = lp.repeat(1, 1, x.shape[2], x.shape[3])
            rp_repeat = rp.repeat(1, 1, x.shape[2], x.shape[3])
            x = self.refine(torch.cat([x, beta_repeat], dim=1))

            if i == 0:
                Q_, P_ = self.Retinex_d(x)

            if i != 0:

                P_ = self.P(x, L, R, betat_repeat)
                Q_ = self.Q(x, P_, L, gamma_repeat,Q_,i)
            R = self.denoiser(torch.cat([P_, rp_repeat], dim=1))
            L = self.Retinexl(torch.cat([Q_, lp_repeat, R], dim=1))  #

            if i == (self.num_iterations-1):
                L_list.append(L)
                L_list.append(R)
            R = torch.clamp(R, min=0, max=10)
            L = torch.clamp(L, min=0, max=1)
            z = R * L
            z = torch.clamp(z, min=0, max=1)
            if i < self.num_iterations - 1:
                z = shift_3d(z)
        return z[:, :, :, 0:256], L_list





