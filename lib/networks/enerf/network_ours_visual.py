import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .feature_net import FeatureNet
from .cost_reg_net import CostRegNet, MinCostRegNet
from . import utils
from lib.config import cfg
from .nerf import NeRF
from itertools import combinations
import time
import json, os

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()

        with open(os.path.join(cfg.result_dir, f'mcp_outputs.json'), 'r') as f:
            self.mcp_outputs = json.load(f)

        for i in range(cfg.enerf.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i))))
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}', nerf_l)

    def render_rays(self, rays, **kwargs):
        level, batch, im_feat, feat_volume, nerf_model = kwargs['level'], kwargs['batch'], kwargs['im_feat'], kwargs['feature_volume'], kwargs['nerf_model']
        world_xyz, uvd, z_vals = utils.sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)
        B, N_rays, N_samples = world_xyz.shape[:3]
        rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[level])
        up_feat_scale = cfg.enerf.cas_config.render_scale[level] / cfg.enerf.cas_config.im_ibr_scale[level]
        if up_feat_scale != 1.:
            B, S, C, H, W = im_feat.shape
            im_feat = F.interpolate(im_feat.reshape(B*S, C, H, W), None, scale_factor=up_feat_scale, align_corners=True, mode='bilinear').view(B, S, C, int(H*up_feat_scale), int(W*up_feat_scale))

        img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
        H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
        B, H, W = len(uvd), int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])
        uvd[..., 0], uvd[..., 1] = (uvd[..., 0]) / (W-1), (uvd[..., 1]) / (H-1)
        
        vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feat_volume)
        img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, level) # B * N * S * (8+3+4)
        
        net_output = nerf_model(vox_feat, img_feat_rgb_dir)
        net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
        
        with torch.no_grad():
            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device)
            mask = utils.mask_viewport(world_xyz, kwargs['batch']['src_exts'], kwargs['batch']['src_ixts'], inv_scale)
            mask = mask.reshape(B, -1, N_samples)
            # mask = torch.ones_like(mask)
        
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
        

        # Progressive masks
        import os, imageio
        os.makedirs('./masks', exist_ok=True)
        os.makedirs('./masks_progressive', exist_ok=True)
        # volume render masks
        H, W = batch['src_inps'].shape[-2:]
        masks_CVs = torch.zeros((N_CV, H, W, 3)).to(masks.device)
        for i in range(N_CV):
            masks_i = utils.raw2outputs_blend(masks[:, i:i+1, ..., None].repeat(1, 1, 1, 1, 4), masks[:, i:i+1], z_vals[:, i:i+1], cfg.enerf.white_bkgd)['rgb']
            masks_i = masks_i.reshape(H, W, 3)
            masks_CVs[i] = masks_i
        
        for i in range(N_CV):
            file_name = f'./masks/{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].item()}_{i}.png'
            imageio.imwrite(file_name, masks_CVs.detach().cpu().numpy()[i])
        
        prev_masks = torch.ones_like(masks_CVs[0])
        masks_progressive = torch.zeros_like(masks_CVs[0])
        for i in range(N_CV):
            masks_progressive += prev_masks * masks_CVs[i]
            prev_masks *= (1 - masks_CVs[i])
            file_name = f'./masks_progressive/{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].item()}_{i}.png'
            imageio.imwrite(file_name, masks_progressive.detach().cpu().numpy())
        # Progressive masks end
        
        # Progressive blending
        # ret = {}
        # H, W = batch['src_inps'].shape[-2:]
        # for i in range(N_CV):
        #     masks_sum = masks[:, :i+1].sum(1)
        #     masks_tmp = torch.where(masks_sum > 0, masks[:, :i+1] / masks_sum, 1 / (i+1))
        #     volume_render_outputs = utils.raw2outputs_blend(net_outputs[:, :i+1], masks_tmp, z_vals[:, :i+1], cfg.enerf.white_bkgd)['rgb']
        #     volume_render_outputs = volume_render_outputs.reshape(-1, H, W, 3)
        #     ret.update({f'rgb_view{i}': volume_render_outputs})
        # # save results to ./rgb_progressive
        # import os, imageio
        # os.makedirs('./rgb_progressive', exist_ok=True)
        # for i in range(N_CV):
        #     file_name = f'./rgb_progressive/{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].item()}_{i}.png'
        #     imageio.imwrite(file_name, ret[f'rgb_view{i}'].detach().cpu().numpy()[0])
        # Progressive blending end
        
        masks_sum = masks.sum(1)
        masks = torch.where(masks_sum > 0, masks / masks_sum, 1 / N_CV)
                    
        volume_render_outputs = utils.raw2outputs_blend(net_outputs, masks, z_vals, cfg.enerf.white_bkgd)
        # store every rgb
        # volume_render_outputs.update({'N_CV': N_CV})
        # volume_render_outputs.update({'rgb_view{}'.format(i): outputs[i] for i in range(N_CV)})
        return volume_render_outputs

    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats  

    def forward(self, batch):        
        N_views = batch['all_src_inps'].shape[1]
                
        selected_views = torch.from_numpy(np.array(list(combinations(range(N_views), 3))))
        
        last = cfg.enerf.cas_config.num - 1
        
        # {scene}_{tar_view}: list -> {k_best_level{last}}: list
        k_best = {}
        k_best.update({f'k_best_level{last}': self.mcp_outputs[f'{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].item()}']})
        # print(k_best)
        
        N_CV = len(k_best[f'k_best_level{last}'])
        
        depth, std, near_far = [None]*N_CV, [None]*N_CV, [None]*N_CV
        feats = self.forward_feat(batch['all_src_inps'])
        
        volume_render_ret = {}
        for i in range(cfg.enerf.cas_config.num):
            # print(k_best[f'k_best_level{last}'])            
            selected_views_i = selected_views[k_best[f'k_best_level{last}']]
            mlp_level_ret = {}
            for v, views in enumerate(selected_views_i):
                batch['src_inps'] = batch['all_src_inps'][:, views]
                batch['src_exts'] = batch['all_src_exts'][:, views]
                batch['src_ixts'] = batch['all_src_ixts'][:, views]
                feature_volume, depth_values, near_far[v] = utils.build_feature_volume(
                        feats[f'level_{i}'][:, views],
                        batch,
                        D=cfg.enerf.cas_config.volume_planes[i],
                        depth=depth[v],
                        std=std[v],
                        near_far=near_far[v],
                        level=i)
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                depth[v], std[v] = utils.depth_regression(depth_prob, depth_values, i, batch)
                if not cfg.enerf.cas_config.render_if[i]:
                    continue
                rays = utils.build_rays(depth[v], std[v], batch, self.training, near_far[v], i)
                # UV(2) +  ray_o (3) + ray_d (3) + ray_near_far (2) + volume_near_far (2)
                im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
                mlp_view_ret = self.batchify_rays_for_mlp(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'][:, views],
                        nerf_model=getattr(self, f'nerf_{i}'),
                        level=i)
                mlp_level_ret.update({key+f'_view{v}': mlp_view_ret[key] for key in mlp_view_ret})
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            volume_render_level_ret = self.merge_mlp_outputs(mlp_level_ret, batch, N_CV)
            
            if cfg.enerf.cas_config.depth_inv[i]:
                volume_render_level_ret.update({'depth_mvs': 1./depth[0]})
            else:
                volume_render_level_ret.update({'depth_mvs': depth[0]})
            volume_render_level_ret.update({'std': std[0]})
            if volume_render_level_ret['rgb'].isnan().any():
                __import__('ipdb').set_trace()
            volume_render_ret.update({key+f'_level{i}': volume_render_level_ret[key] for key in volume_render_level_ret})
            
        return volume_render_ret
