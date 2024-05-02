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

class MCP(nn.Module):
    def __init__(self,):
        super(MCP, self).__init__()
        self.feature_net = FeatureNet()
        for i in range(cfg.enerf.cas_config.num):
            if i == 0:
                cost_reg_l = MinCostRegNet(int(32 * (2**(-i))))
            else:
                cost_reg_l = CostRegNet(int(32 * (2**(-i))))
            setattr(self, f'cost_reg_{i}', cost_reg_l)
            nerf_l = NeRF(feat_ch=cfg.enerf.cas_config.nerf_model_feat_ch[i]+3)
            setattr(self, f'nerf_{i}', nerf_l)

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
    
    def calc_mask(self, src_views_id, cam_tar, ray_tar, batch):
        batch['src_inps'] = batch['all_src_inps'][:, src_views_id]
        batch['src_exts'] = batch['all_src_exts'][:, src_views_id]
        batch['src_ixts'] = batch['all_src_ixts'][:, src_views_id]
        feats = self.forward_feat(batch['src_inps'])
        depth, std, near_far = None, None, None
        
        mask_all = {}
        for i in range(cfg.enerf.cas_config.num):
            feature_volume, depth_values, near_far = utils.build_feature_volume(
                    feats[f'level_{i}'],
                    batch,
                    D=cfg.enerf.cas_config.volume_planes[i],
                    depth=depth,
                    std=std,
                    near_far=near_far,
                    level=i)
            
            feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
            depth, std = utils.depth_regression(depth_prob, depth_values, i, batch)
            if not cfg.enerf.cas_config.render_if[i]:
                continue
        
            ray_tar = utils.build_rays(depth, std, batch, self.training, near_far, i)
            N_samples = cfg.enerf.cas_config.num_samples[i]
            world_xyz, uvd, z_vals = utils.sample_along_depth(ray_tar, N_samples=N_samples, level=i)
            
            B, S, _, H_O, W_O = batch['src_inps'].shape # B, S, C, H, W
            H, W = int(H_O * cfg.enerf.cas_config.render_scale[i]), int(W_O * cfg.enerf.cas_config.render_scale[i])
            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=world_xyz.device)
            
            im_feat = feats[f'level_{cfg.enerf.cas_config.render_im_feat_level[i]}']
            rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[i])
            img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
            vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feature_volume)
            img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, i) # B * N * S * (8+3+4)
            net_output = getattr(self, f'nerf_{i}')(vox_feat, img_feat_rgb_dir)
            net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
            
            mask = utils.mask_viewport(world_xyz, batch['src_exts'], batch['src_ixts'], inv_scale)
            mask = mask.reshape(B, -1, N_samples, 1) / N_samples
            # mask = net_output[..., -1:]
            # mask.mul_(net_output[..., -1:])
            mask = mask.repeat(1, 1, 1, 4) # rgb + alpha
            mask = utils.raw2outputs(mask, z_vals, cfg.enerf.white_bkgd)['rgb'].mean(-1)
            mask = mask.reshape(B, H, W)
            
            # LPIPS check
            # import torch.nn.functional as F
            # std = F.interpolate(std[None], (H, W), mode='bilinear', align_corners=True)[0]
            # # take gt image
            # gt = batch['rgb_{}'.format(i)]
            # pred = utils.raw2outputs(net_output, z_vals, cfg.enerf.white_bkgd)['rgb']
            # gt = gt.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            # pred = pred.reshape(B, H, W, 3).permute(0, 3, 1, 2)
            # import lpips
            # loss_fn = lpips.LPIPS(net='alex', spatial=True).to(pred.device)
            # d = loss_fn.forward(gt, pred)
            # # save d as an image to ./lpips_dist
            # # and concat together rgb, d, std and mask
            # import os
            # import imageio
            # os.makedirs('./lpips_dist', exist_ok=True)
            # scene_info = f'{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"].detach().cpu().numpy().tolist()}'
            # file_name = f'./lpips_dist/{scene_info}_{src_views_id}_{i}.png'
            # output = utils.raw2outputs(net_output, z_vals, cfg.enerf.white_bkgd)['rgb']
            # rgb = output.detach().cpu().numpy().reshape(H, W, 3)
            # rgb = (rgb * 255).astype(np.uint8)
            # std_rgb = std.detach().cpu().numpy().reshape(H, W, 1).repeat(3, axis=-1)
            # std_rgb = (std_rgb - std_rgb.min()) / (std_rgb.max() - std_rgb.min())
            # std_rgb = (std_rgb * 255).astype(np.uint8)
            # mask_rgb = mask.detach().cpu().numpy().reshape(H, W, 1).repeat(3, axis=-1)
            # mask_rgb = (mask_rgb * 255).astype(np.uint8)
            # # save rgb, d, std and mask
            # d_rgb = (d[0, 0] * 255).detach().cpu().numpy().astype(np.uint8)
            # d_rgb = np.stack([d_rgb, d_rgb, d_rgb], axis=-1)
            # rgb = np.concatenate((rgb, d_rgb, std_rgb, mask_rgb), axis=1)
            # imageio.imwrite(file_name, rgb)
            
            mask_all.update({f'mask_level{i}': mask})
        
        return mask_all
    
    def search_k_best_views(self, batch, masks, k, prev_mask, results, level):
        if k == 0:
            return results
        H, W = masks[f'mask_level{level}_view0'].shape[-2:]
        max_update_ratio, best_mask_id = 0, None
        for i in range(len(masks)):
            if i in results:
                continue
            mask = masks[f'mask_level{level}_view{i}']
            update_ratio = (mask * prev_mask).sum() / (H*W)
            if update_ratio > max_update_ratio:
                max_update_ratio = update_ratio
                best_mask_id = i
        
        if best_mask_id is None:
            return results
        
        # print(f'level {level}, best mask id: {best_mask_id}, update: {update_ratio}')
            
        prev_mask = prev_mask * (1 - masks[f'mask_level{level}_view{best_mask_id}'])
        # prev_mask = prev_mask + masks[f'mask_level{level}_view{best_mask_id}']
        results.append(best_mask_id)
        
        return self.search_k_best_views(batch, masks, k-1, prev_mask, results, level)            

    def forward(self, batch):        
        N_views = batch['all_src_inps'].shape[1]
                
        selected_views = torch.from_numpy(np.array(list(combinations(range(N_views), 3))))
        # st = time.time()
        
        # mx, mn = None, None
        src_masks_all = {}
        for i, src_views_id in enumerate(selected_views):
            src_masks = self.calc_mask(src_views_id, batch['tar_ext'], batch['tar_ixt'], batch)
            src_masks_all.update({key+f'_view{i}': src_masks[key] for key in src_masks})
                
        # print(f'mask time: {time.time()-st}')
        
        k_best = {}
        for i in range(cfg.enerf.cas_config.num):
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            prev_mask = torch.ones_like(src_masks_all[f'mask_level{i}_view0'])
            results = []
            src_masks_leveli = {key: src_masks_all[key] for key in src_masks_all if key.startswith(f'mask_level{i}')}
            k_best_i = self.search_k_best_views(batch, src_masks_leveli, cfg.enerf.cas_config.k_best, prev_mask, results, level=i)
            k_best_i = torch.tensor(k_best_i, dtype=torch.long)
            # k_best.update({f'k_best_level{i}': k_best_i})
            key = f'{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"][0]}'
            val = k_best_i.detach().cpu().numpy().tolist()
            k_best.update({key: val})
        return k_best