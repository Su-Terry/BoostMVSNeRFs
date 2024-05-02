from lib.config import cfg
from lib.networks.enerf.utils import *
from itertools import combinations
import os
import numpy as np

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.nn import functional as F
from .utils import homo_warp, get_ndc_coordinate
from .renderer import gen_dir_feature, gen_pts_feats
from inplace_abn import InPlaceABN
from .renderer import run_network_mvs


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        self.freq_bands = freq_bands.reshape(1,-1,1).cuda()

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        repeat = inputs.dim()-1
        inputs_scaled = (inputs.unsqueeze(-2) * self.freq_bands.view(*[1]*repeat,-1,1)).reshape(*inputs.shape[:-1],-1)
        inputs_scaled = torch.cat((inputs, torch.sin(inputs_scaled), torch.cos(inputs_scaled)),dim=-1)
        return inputs_scaled

def get_embedder(multires, i=0, input_dims=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : input_dims,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn



class Renderer_ours(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_ours, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward_alpha(self, x):

        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = torch.relu(self.alpha_linear(h))
        return alpha


    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class Renderer_color_fusion(nn.Module):
    def __init__(self, D=8, W=128, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4],use_viewdirs=False):
        """
        """
        super(Renderer_color_fusion, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=True)] + [
                nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                range(D - 1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)

        attension_dim = 16 + 3 + self.in_ch_views//3 #  16 + rgb dim + angle dim
        self.ray_attention = MultiHeadAttention(4, attension_dim, 4, 4)

        if use_viewdirs:
            self.feature_linear = nn.Sequential(nn.Linear(W, 16), nn.ReLU())
            self.alpha_linear = nn.Sequential(nn.Linear(W, 1), nn.ReLU())
            self.rgb_out = nn.Sequential(nn.Linear(attension_dim, 3),nn.Sigmoid())  #
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_out.apply(weights_init)

    def forward_alpha(self,x):
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha


    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim - self.in_ch_pts - self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)

        # color
        input_views = input_views.reshape(-1, 3, self.in_ch_views//3)
        rgb = input_feats[..., 8:].reshape(-1, 3, 4)
        rgb_in = rgb[..., :3]

        N = rgb.shape[0]
        feature = self.feature_linear(h)
        h = feature.reshape(N, 1, -1).expand(-1, 3, -1)
        h = torch.cat((h, input_views, rgb_in), dim=-1)
        h, _ = self.ray_attention(h, h, h, mask=rgb[...,-1:])
        rgb = self.rgb_out(h)

        rgb = torch.sum(rgb , dim=1).reshape(*alpha.shape[:2], 3)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

class Renderer_attention2(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_attention, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.attension_dim = 4 + 8
        self.color_attention = MultiHeadAttention(4, self.attension_dim, 4, 4)
        self.weight_out = nn.Linear(self.attension_dim, 3)



        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + self.in_ch_pts, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(11, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, x):
        N_ray, N_sample, dim = x.shape
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        if input_feats.shape[-1]>8+3:
            colors = input_feats[...,8:].view(N_ray*N_sample,-1,4)
            weight = torch.cat((colors,input_feats[...,:8].reshape(N_ray*N_sample, 1, -1).expand(-1, colors.shape[-2], -1)),dim=-1)

            weight, _ = self.color_attention(weight, weight, weight)
            colors = torch.sum(self.weight_out(weight),dim=-2).view(N_ray, N_sample, -1)

            # colors = self.weight_out(input_feats)

        else:
            colors = input_feats[...,-3:]

        h = input_pts
        # bias = self.pts_bias(colors)
        bias = self.pts_bias(torch.cat((input_feats[...,:8],colors),dim=-1))
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) * bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        outputs = torch.cat((outputs,colors), dim=-1)
        return outputs

class Renderer_attention(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_attention, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.attension_dim = 4 + 8
        self.color_attention = MultiHeadAttention(4, self.attension_dim, 4, 4)
        self.weight_out = nn.Linear(self.attension_dim, 3)

        # self.weight_out = nn.Linear(self.in_ch_feat, 8)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.in_ch_pts, W, bias=True)] + [nn.Linear(W, W, bias=True)]*(D-1))
        self.pts_bias = nn.Linear(11, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward(self, x):
        N_ray, N_sample, dim = x.shape
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        if input_feats.shape[-1]>8+3:
            colors = input_feats[...,8:].view(N_ray*N_sample,-1,4)
            weight = torch.cat((colors,input_feats[...,:8].reshape(N_ray*N_sample, 1, -1).expand(-1, colors.shape[-2], -1)),dim=-1)

            weight, _ = self.color_attention(weight, weight, weight)
            colors = torch.sum(torch.sigmoid(self.weight_out(weight)),dim=-2).view(N_ray, N_sample, -1)

            # colors = self.weight_out(input_feats)

        else:
            colors = input_feats[...,-3:]

        h = input_pts
        # bias = self.pts_bias(colors)
        bias = self.pts_bias(torch.cat((input_feats[...,:8],colors),dim=-1))
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) + bias
            h = F.relu(h)
            # if i in self.skips:
            #     h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha, colors], -1)
        else:
            outputs = self.output_linear(h)
        outputs = torch.cat((outputs,colors), dim=-1)
        return outputs

class Renderer_linear(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, input_ch_feat=8, skips=[4], use_viewdirs=False):
        """
        """
        super(Renderer_linear, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.in_ch_pts, self.in_ch_views, self.in_ch_feat = input_ch, input_ch_views, input_ch_feat

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W, bias=True)] + [nn.Linear(W, W, bias=True) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        self.pts_bias = nn.Linear(input_ch_feat, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.pts_linears.apply(weights_init)
        self.views_linears.apply(weights_init)
        self.feature_linear.apply(weights_init)
        self.alpha_linear.apply(weights_init)
        self.rgb_linear.apply(weights_init)

    def forward_alpha(self,x):
        dim = x.shape[-1]
        input_pts, input_feats = torch.split(x, [self.in_ch_pts, self.in_ch_feat], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) + bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        return alpha

    def forward(self, x):
        dim = x.shape[-1]
        in_ch_feat = dim-self.in_ch_pts-self.in_ch_views
        input_pts, input_feats, input_views = torch.split(x, [self.in_ch_pts, in_ch_feat, self.in_ch_views], dim=-1)

        h = input_pts
        bias = self.pts_bias(input_feats) #if in_ch_feat == self.in_ch_feat else  input_feats
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h) + bias
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)


        if self.use_viewdirs:
            alpha = torch.relu(self.alpha_linear(h))
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = torch.sigmoid(self.rgb_linear(h))
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

class MVSNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch_pts=3, input_ch_views=3, input_ch_feat=8, skips=[4], net_type='v2'):
        """
        """
        super(MVSNeRF, self).__init__()

        self.in_ch_pts, self.in_ch_views,self.in_ch_feat = input_ch_pts, input_ch_views, input_ch_feat

        # we provide two version network structure
        if 'v0' == net_type:
            self.nerf = Renderer_ours(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)
        elif 'v1' == net_type:
            self.nerf = Renderer_attention(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)
        elif 'v2' == net_type:
            self.nerf = Renderer_linear(D=D, W=W,input_ch_feat=input_ch_feat,
                     input_ch=input_ch_pts, output_ch=4, skips=skips,
                     input_ch_views=input_ch_views, use_viewdirs=True)

    def forward_alpha(self, x):
        return self.nerf.forward_alpha(x)

    def forward(self, x):
        RGBA = self.nerf(x)
        return RGBA

def create_nerf_mvs(args, pts_embedder=True, use_mvs=False, dir_embedder=True):
    """Instantiate mvs NeRF's MLP model.
    """

    if pts_embedder:
        embed_fn, input_ch = get_embedder(args.multires, args.i_embed, input_dims=args.pts_dim)
    else:
        embed_fn, input_ch = None, args.pts_dim

    embeddirs_fn = None
    if dir_embedder:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed, input_dims=args.dir_dim)
    else:
        embeddirs_fn, input_ch_views = None, args.dir_dim


    skips = [4]
    model = MVSNeRF(D=args.netdepth, W=args.netwidth,
                 input_ch_pts=input_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_feat=args.feat_dim, net_type=args.net_type).to(device)

    grad_vars = []
    grad_vars += list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = MVSNeRF(D=args.netdepth, W=args.netwidth,
                 input_ch_pts=input_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_feat=args.feat_dim).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda pts, viewdirs, rays_feats, network_fn: run_network_mvs(pts, viewdirs, rays_feats, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    EncodingNet = None
    if use_mvs:
        EncodingNet = MVSNet().to(device)
        grad_vars += list(EncodingNet.parameters())    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    start = 0


    ##########################

    # Load checkpoints
    ckpts = []
    if args.ckpt is not None and args.ckpt != 'None':
        ckpts = [args.ckpt]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 :
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # Load model
        if use_mvs:
            state_dict = ckpt['network_mvs_state_dict']
            EncodingNet.load_state_dict(state_dict)

        model.load_state_dict(ckpt['network_fn_state_dict'])
        # if model_fine is not None:
        #     model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'network_mvs': EncodingNet,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }


    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#############################################     MVS Net models        ################################################
class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        # self.bn = nn.ReLU()

    def forward(self, x):
        return self.bn(self.conv(x))

###################################  feature net  ######################################
class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.conv0(x) # (B, 8, H, W)
        x = self.conv1(x) # (B, 16, H//2, W//2)
        x = self.conv2(x) # (B, 32, H//4, W//4)
        x = self.toplayer(x) # (B, 32, H//4, W//4)

        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x

# class MVSNet(nn.Module):
class Network(nn.Module):
    def __init__(self,
                 num_groups=1,
                 norm_act=InPlaceABN,
                 levels=1):
        super(Network, self).__init__()
        self.levels = levels  # 3 depth levels
        self.n_depths = [128,32,8]
        self.G = num_groups  # number of groups in groupwise correlation

        self.N_importance = 0
        self.chunk = 1024

        self.feature = FeatureNet()
        self.cost_reg_2 = CostRegNet(32+9, norm_act)
        
        # self.nerf = MVSNeRF(D=6, W=128,
        #          input_ch_pts=3, skips=[4],
        #          input_ch_views=3, input_ch_feat=8, net_type='v0')
        # from lib.networks.enerf.nerf import NeRF as mvsnerf
        # self.nerf = mvsnerf(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[0]+3)
        self.nerf = MVSNeRF(D=6, W=128,
                    input_ch_pts=63, skips=[4],
                    input_ch_views=3, input_ch_feat=20, net_type='v0')


    # def __init__(self,):
    #     super(Network, self).__init__()
    #     from lib.networks.enerf.feature_net import FeatureNet
    #     self.feature = FeatureNet()

    #     # with open(os.path.join(cfg.result_dir, f'mcp_outputs.json'), 'r') as f:
    #     #     self.mcp_outputs = json.load(f)

    #     from lib.networks.enerf.cost_reg_net import CostRegNet, MinCostRegNet
    #     for i in range(cfg.enerf.cas_config.num):
    #         if i == 0:
    #             cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
    #         else:
    #             cost_reg_l = CostRegNet(int(32 * (2**(-i))))
    #         setattr(self, f'cost_reg_{i}', cost_reg_l)
    #         nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
    #         setattr(self, f'nerf_{i}', nerf_l)


    def build_volume_costvar(self, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, 1, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_feat, proj_mat) in enumerate(zip(src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks += in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / in_masks
        img_feat = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks

    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        img_feat = torch.empty((B, 9 + 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks

#####################################################################################################################
    def ray_marcher(self, rays, N_sample, lindisp=False, perturb=0, bbox_3D=None):
        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        # near, far = near_far.min() * torch.ones_like(rays[..., :1]), near_far.max() * torch.ones_like(rays[..., :1])
        near, far = rays[..., 6:7], rays[..., 7:8]

        zsteps = torch.linspace(0., 1., steps=N_sample, device=rays.device)  # (N_sample)
        if not lindisp:
            z_vals = near * (1. - zsteps) + far * zsteps  # (N_sample)
        else:
            z_vals = 1. / (1. / near * (1. - zsteps) + 1. / far * zsteps)
        z_vals = z_vals.expand(rays.shape[0], N_sample)  # (N_rays, N_sample)

        xyz_coarse_sampled = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_sample, 3)

        return xyz_coarse_sampled[None], z_vals[None]
    
    
    def run_network_mvs(self, pts, viewdirs, alpha_feat, embed_fn, embeddirs_fn):
        """Prepares inputs and applies network 'fn'.
        """

        if embed_fn is not None:
            pts = embed_fn(pts)

        if alpha_feat is not None:
            pts = torch.cat((pts,alpha_feat), dim=-1)

        if viewdirs is not None:
            if viewdirs.dim()!=3:
                viewdirs = viewdirs[:, None].expand(-1,pts.shape[1],-1)

            if embeddirs_fn is not None:
                viewdirs = embeddirs_fn(viewdirs)
            pts = torch.cat([pts, viewdirs], -1)

        return pts
    
    def rendering(self, batch, rays_pts, rays_ndc, depth_candidates,rays_o, rays_dir, volume_feature):
        cos_angle = torch.norm(rays_dir, dim=-1)

        angle = gen_dir_feature(batch['src_exts'][0][0], rays_dir/cos_angle.unsqueeze(-1))
        # print(angle.min(), angle.max())
        pose_ref = {'w2cs': batch['src_exts'][0], 'intrinsics': batch['src_ixts'][0]}
        
        rgbs = unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[0])
        # rgbs = batch['src_inps']
        # print(rgbs.min(), rgbs.max())
        input_feat  = gen_pts_feats(rgbs, volume_feature, rays_pts, pose_ref, rays_ndc, 20, None)

        embed_fn, _ = get_embedder(10, 0, input_dims=3)
        embeddirs_fn = None
        # print(rays_ndc.shape, angle.shape, input_feat.shape)
        raw = self.run_network_mvs(rays_ndc[0], angle[0], input_feat, embed_fn, embeddirs_fn)
        
        return raw

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        # world_xyz, uvd, z_vals = sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)

        world_xyz, z_vals = self.ray_marcher(rays[0], cfg.enerf.cas_config.num_samples[level])

        B, N_rays, N_samples = world_xyz.shape[:3]
        chunk = N_rays // 10
        net_output = torch.zeros(B, N_rays, N_samples, 4, device=world_xyz.device)
        mask = torch.zeros(B, N_rays, N_samples, device=world_xyz.device)

        for i in range(0, N_rays, chunk):
            world_xyz_i = world_xyz[:, i:i + chunk]
            # uvd_i = uvd[:, i:i + chunk]
            rays_i = rays[:, i:i + chunk]
            # z_vals_i = z_vals[:, i:i + chunk]
            # rgbs = unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[level])
            # up_feat_scale = cfg.enerf.cas_config.render_scale[level] / cfg.enerf.cas_config.im_ibr_scale[level]
            # if up_feat_scale != 1. and i == 0:
            #     B, S, C, H, W = im_feat.shape
            #     im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))
            # # img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
            H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
            B, H, W = 1, int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])
            # uvd_i[..., 0], uvd_i[..., 1] = (uvd_i[..., 0]) / (W-1), (uvd_i[..., 1]) / (H-1)

            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device)
            
            # print(batch['src_exts'][0][0])
            uvd_i = get_ndc_coordinate(batch['src_exts'][0][0], batch['src_ixts'][0][0], world_xyz_i[0], inv_scale, near=batch['near_far'].min(), far=batch['near_far'].max(), pad=24)[None]
            raw = self.rendering(batch, world_xyz_i[0], uvd_i, z_vals, rays_i[..., :3], rays_i[..., 3:6], feat_volume)
            # vox_feat = get_vox_feat(uvd_i.reshape(B, -1, 3), feat_volume)
            # img_feat_rgb_dir = get_img_feat(world_xyz_i, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
            # net_output_i = nerf_model(vox_feat, img_feat_rgb_dir) 

            net_output_i = nerf_model(raw)
            # print(net_output_i.shape)
            net_output_i = net_output_i.reshape(B, -1, N_samples, net_output_i.shape[-1])
        
            with torch.no_grad():
                inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device)
                mask_i = mask_viewport(world_xyz_i, kwargs['batch']['src_exts'], kwargs['batch']['src_ixts'], inv_scale)
                mask_i = mask_i.reshape(B, -1, N_samples) #/ N_samples
                # mask_i = torch.ones_like(mask_i)

            net_output[:, i:i + chunk] = net_output_i
            mask[:, i:i + chunk] = mask_i
        
        outputs = {
            'net_output': net_output,
            'z_vals': z_vals,
            'mask': mask
        }
        return outputs

    def batchify_rays_for_mlp(self, rays, **kwargs):
        all_ret = {}
        chunk = cfg.enerf.chunk_size
        for i in range(0, rays.shape[1], chunk):
            ret = self.render_rays(rays[:, i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=1) for k in all_ret}
        return all_ret

    def merge_mlp_outputs(self, outputs, batch, N_CV):
        # print(outputs.keys())
        # assert (outputs[f'net_output_view0'] != outputs[f'net_output_view1']).any()
        net_outputs = torch.stack([outputs[f'net_output_view{i}'] for i in range(N_CV)], dim=1)
        masks = torch.stack([outputs[f'mask_view{i}'] for i in range(N_CV)], dim=1)
        z_vals = torch.stack([outputs[f'z_vals_view{i}'] for i in range(N_CV)], dim=1)
        
        # outputs = []
        # for i in range(N_CV):
        #     outputs.append(raw2outputs_blend(net_outputs[:, i:i+1], torch.ones_like(masks[:, i:i+1]), z_vals[:, i:i+1], cfg.enerf.white_bkgd)['rgb'])
        
        # prev_masks = torch.ones_like(masks)
        # for i in range(N_CV):
        #     masks[:, i] = masks[:, i] * prev_masks[:, i]
        #     prev_masks[:, i] = prev_masks[:, i] * (1 - masks[:, i])
        masks_sum = masks.sum(1)
        masks = torch.where(masks_sum > 0, masks / masks_sum, 1 / N_CV)
                    
        volume_render_outputs = raw2outputs_blend(net_outputs, masks, z_vals, cfg.enerf.white_bkgd)
        # store every rgb
        # volume_render_outputs.update({'N_CV': N_CV})
        # volume_render_outputs.update({'rgb_view{}'.format(i): outputs[i] for i in range(N_CV)})
        
        # only store the rgb
        volume_render_outputs = {'rgb': volume_render_outputs['rgb']}
        return volume_render_outputs

#####################################################################################################################

    def get_proj_mats(self, batch):
        proj_mats = []
        for i, (src_ext, src_ixt) in enumerate(zip(batch['src_exts'][0], batch['src_ixts'][0])):
            proj_mat_l = torch.eye(4)
            src_ixt[:2] *= 0.25
            # print(src_ixt.shape, src_ext.shape)
            # print(src_ixt)
            # print(src_ixt)
            proj_mat_l[:3, :4] = src_ixt @ src_ext[:3, :4]
            src_ixt[:2] *= 4
            if i==0:
                ref_proj_inv = torch.inverse(proj_mat_l)
                proj_mats += [torch.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]

        # print(torch.from_numpy(np.stack(proj_mats)[:, :3]).float().unsqueeze(0))

        return torch.from_numpy(np.stack(proj_mats)[:, :3]).float().unsqueeze(0)

    # def forward(self, imgs, proj_mats, near_far, tar_view, scene_name, all_scenes, pad=0,  return_color=False, lindisp=False):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)
    def forward(self, batch):
        import json
        with open(os.path.join(cfg.result_dir, f'mcp_outputs.json'), 'r') as f:
            self.mcp_outputs = json.load(f)

        N_views = batch['all_src_inps'].shape[1]
                
        selected_views = torch.from_numpy(np.array(list(combinations(range(N_views), 3))))
        
        last = 1
        
        k_best = {}
        k_best.update({f'k_best_level{last}': self.mcp_outputs[f'{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].item()}']})

        print(N_views, k_best)
        
        D = cfg.enerf.cas_config.num_samples[last-1]
        N_CV = len(k_best[f'k_best_level{last}'])
        # N_CV = 1
        # near_far = batch['near_far'].to(batch['all_src_inps'].device)
        # near = near_far.min()
        # far = near_far.max()

        
        # depth, std, near_far = [None]*N_CV, [None]*N_CV, [None]*N_CV
        feats = self.feature(batch['all_src_inps'][0]) # (B*V, C, H, W)
        feats = feats.view(1, -1, *feats.shape[1:])  # (B, V, C, h, w)
        
        # volume_render_ret = {}
        if N_views == 6:
            selected_views_i = selected_views[k_best[f'k_best_level{last}']]
        else:
            selected_views_i = [[0,1,2], [0, 1, 3], [0, 2, 3]]
        mlp_level_ret = {}
        for v, views in enumerate(selected_views_i):
            # if v > 0:
                # break
            # views = torch.tensor([0, 1, 2]).to(batch['all_src_inps'].device)
            t_vals = torch.linspace(0., 1., steps=D, device=batch['all_src_inps'].device, dtype=batch['all_src_inps'].dtype)  # (B, D)
            near, far = (batch['depth_ranges'][0, views].min()*0.8).to(batch['all_src_inps'].device), (batch['depth_ranges'][0, views].max()*1.2).to(batch['all_src_inps'].device)
            depth_values = near * (1.-t_vals) + far * (t_vals)
            depth_values = depth_values.unsqueeze(0).to(batch['all_src_inps'].device)

            # print(near)
            # print(far)
            # print(batch['depth_ranges'][0, views])
            # print(depth_values)
            batch['near_far'] = torch.stack([near, far]).to(batch['all_src_inps'].device)
            # print(batch['near_far'])
            batch['src_inps'] = batch['all_src_inps'][:, views]
            batch['src_exts'] = batch['all_src_exts'][:, views]
            batch['src_ixts'] = batch['all_src_ixts'][:, views]
            proj_mats = self.get_proj_mats(batch).to(batch['all_src_inps'].device)
            volume_feat, _ = self.build_volume_costvar_img(batch['src_inps'], feats[:, views], proj_mats, depth_values, pad=24)
            volume_feat = self.cost_reg_2(volume_feat) # (B, 1, D, h, w)
            volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])


            # build rays
            depth = (near + far) / 2.
            depth = torch.ones_like(batch['src_inps'][0, 0, 0]) * depth
            std = (far - near) / 2.
            std = torch.ones_like(batch['src_inps'][0, 0, 0]) * std
            depth = depth[None]
            std = std[None]
            near_far = torch.stack([depth-std, depth+std], dim=1)

            # rays = build_rays(depth, std, batch, 0, near_far, 0)
            rays = batch['rays_0']
            # print(rays.shape)

            # batchify rays for mlp
            mlp_view_ret = self.batchify_rays_for_mlp(rays, level=0, batch=batch, im_feat=feats[:, views], feature_volume=volume_feat, nerf_model=self.nerf)

            mlp_level_ret.update({key+f'_view{v}': mlp_view_ret[key] for key in mlp_view_ret})

        volume_rendered_ret = self.merge_mlp_outputs(mlp_level_ret, batch, N_CV)
        # from lib.networks.enerf.utils import raw2outputs
        # volume_rendered_ret = raw2outputs(mlp_level_ret['net_output_view0'], mlp_level_ret['z_vals_view0'], cfg.enerf.white_bkgd)
        volume_render_ret = {}
        volume_render_ret.update({key+f'_level0': volume_rendered_ret[key] for key in volume_rendered_ret})
        
        # h, w = batch['meta'][f'h_0'], batch['meta'][f'w_0']
        # H, W = batch['src_inps'].shape[-2:]
        # H, W = int(H * cfg.enerf.cas_config.render_scale[0]), int(W * cfg.enerf.cas_config.render_scale[0])
        # volume_render_ret['rgb_level0'] = volume_render_ret['rgb_level0'].reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        # volume_render_ret['rgb_level0'] = F.interpolate(volume_render_ret['rgb_level0'], (h, w), mode='bilinear')
        # B = 1
        # volume_render_ret['rgb_level0'] = volume_render_ret['rgb_level0'].permute(0, 2, 3, 1).reshape(B, -1, 3)


        return volume_render_ret

        # B, V, _, H, W = imgs.shape

        # imgs = imgs.reshape(B * V, 3, H, W)

        # D = 128

        # ##
        # N_views = 6
        # selected_views = torch.from_numpy(np.array(list(combinations(range(N_views), 3))))

        # # k best
        # k_best = {}
        # k_best.update({f'k_best_level1': self.mcp_outputs[f'{scene_name}_{tar_view}']})
        # N_CV = len(k_best[f'k_best_level1'])

        # selected_views_i = selected_views[k_best[f'k_best_level1']]

        # mlp_level_ret = {}
        # for v, views in enumerate(selected_views_i):
        #     # pass

        #     imgs = all_scenes[views].reshape(B * V, 3, H, W)
        #     feats = self.feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)
        #     imgs = imgs.view(B, V, 3, H, W)
        #     feats_l = feats  # (B*V, C, h, w)
        #     feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)

        #     t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        #     near, far = near_far  # assume batch size==1
        #     if not lindisp:
        #         depth_values = near * (1.-t_vals) + far * (t_vals)
        #     else:
        #         depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        #     depth_values = depth_values.unsqueeze(0)
        #     # volume_feat, in_masks = self.build_volume_costvar(feats_l, proj_mats, depth_values, pad=pad)
        #     volume_feat, in_masks = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad)
            
        #     if return_color:
        #         feats_l = torch.cat((volume_feat[:,:V*3].view(B, V, 3, *volume_feat.shape[2:]),in_masks.unsqueeze(2)),dim=2)


        #     volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        #     volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

        #     mlp_level_ret.update({f'output_view{v}': volume_feat[i] for i in range(len(volume_feat))})

        return volume_feat, feats_l, depth_values


class RefVolume(nn.Module):
    def __init__(self, volume):
        super(RefVolume, self).__init__()

        self.feat_volume = nn.Parameter(volume)

    def forward(self, ray_coordinate_ref):
        '''coordinate: [N, 3]
            z,x,y
        '''

        device = self.feat_volume.device
        H, W = ray_coordinate_ref.shape[-3:-1]
        grid = ray_coordinate_ref.view(-1, 1, H, W, 3).to(device) * 2 - 1.0  # [1 1 H W 3] (x,y,z)
        features = F.grid_sample(self.feat_volume, grid, align_corners=True, mode='bilinear')[:, :, 0].permute(2, 3, 0,1).squeeze()
        return features

