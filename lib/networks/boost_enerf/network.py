import torch
from torch.nn import functional as F
from lib.networks.enerf import utils
from lib.config import cfg
from lib.networks.enerf import network
import json
import os
import time

class Network(network.Network):
    def __init__(self, preprocess=False):
        super(Network, self).__init__()
                
        if not preprocess:
            view_selection_file = os.path.join(cfg.result_dir, f'view_selection.json')
            # View selection should be preprocessed.
            if not os.path.exists(view_selection_file):
                raise "View selection file not found. Please run view selection first."
            with open(view_selection_file, 'r') as f:
                self.view_selection_outputs = json.load(f)
            
    def calc_mask(self, src_views_id, batch):
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
            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=world_xyz.device).unsqueeze(0).expand(B, -1)
            
            im_feat = feats[f'level_{cfg.enerf.cas_config.render_im_feat_level[i]}']
            rgbs = utils.unpreprocess(batch['src_inps'], render_scale=cfg.enerf.cas_config.render_scale[i])
            img_feat_rgb = torch.cat((im_feat, rgbs), dim=2)
            vox_feat = utils.get_vox_feat(uvd.reshape(B, -1, 3), feature_volume)
            img_feat_rgb_dir = utils.get_img_feat(world_xyz, img_feat_rgb, batch, self.training, i) # B * N * S * (8+3+4)
            net_output = getattr(self, f'nerf_{i}')(vox_feat, img_feat_rgb_dir)
            net_output = net_output.reshape(B, -1, N_samples, net_output.shape[-1])
            
            mask = utils.mask_viewport(world_xyz, batch['src_exts'], batch['src_ixts'], inv_scale)
            mask = mask.reshape(B, -1, N_samples, 1) / N_samples
            mask = mask.repeat(1, 1, 1, 4) # rgb + alpha
            mask = utils.raw2outputs(mask, z_vals, cfg.enerf.white_bkgd)['rgb'].mean(-1)
            mask = mask.reshape(B, H, W)
            
            mask_all.update({f'mask_level{i}': mask})
        
        return mask_all
    
    def search_k_best_views(self, masks, k, level):
        results = []
        prev_mask = torch.ones_like(masks[f'mask_level{level}_view0'])
        H, W = masks[f'mask_level{level}_view0'].shape[-2:]
        for _ in range(k):
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
                break
            
            prev_mask = prev_mask * (1 - masks[f'mask_level{level}_view{best_mask_id}'])
            results.append(best_mask_id)
            
        if results == []:
            results.append(0)
        
        return results
    
    def forward_view_selection(self, batch):
        N_views = batch['all_src_inps'].shape[1]
        selected_views = torch.combinations(torch.arange(N_views), 3)

        # Obtain masks in parallel
        # st = time.time()
        src_masks_all = {}
        for i, src_views_id in enumerate(selected_views):
            src_masks = self.calc_mask(src_views_id, batch)
            src_masks_all.update({key+f'_view{i}': src_masks[key] for key in src_masks})
        # print(f'mask time: {time.time()-st}')
                
        k_best = {}
        for i in range(cfg.enerf.cas_config.num):
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            src_masks_leveli = {key: src_masks_all[key] for key in src_masks_all if key.startswith(f'mask_level{i}')}
            k_best_i = self.search_k_best_views(src_masks_leveli, cfg.enerf.cas_config.k_best, level=i)
            k_best_i = torch.tensor(k_best_i, dtype=torch.long)
            key = f'{batch["meta"]["scene"][0]}_{batch["meta"]["tar_view"][0]}'
            val = k_best_i.detach().cpu().numpy().tolist()
            k_best.update({key: val})
        return k_best
    
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
            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device).unsqueeze(0).expand(B, -1)
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
    
    def merge_mlp_outputs(self, outputs, K):
        net_outputs = torch.stack([outputs[f'net_output_view{i}'] for i in range(K)], dim=1)
        masks = torch.stack([outputs[f'mask_view{i}'] for i in range(K)], dim=1)
        z_vals = torch.stack([outputs[f'z_vals_view{i}'] for i in range(K)], dim=1)
        
        masks_sum = masks.unsqueeze(1).sum(2)
        masks = torch.where(masks_sum > 0, masks / masks_sum, 1 / K)
                    
        volume_render_outputs = utils.raw2outputs_blend(net_outputs, masks, z_vals, cfg.enerf.white_bkgd)
        return volume_render_outputs

    def forward(self, batch):
        B, N = batch['all_src_inps'].shape[:2]
        I = cfg.enerf.cost_volume_input_views
        K = cfg.enerf.cas_config.k_best
        selected_views = torch.combinations(torch.arange(N), I).to(batch['all_src_inps'].device) # (N, 3)
                
        scene_names, target_views = batch["meta"]["scene"], batch["meta"]["tar_view"]
        k_best = torch.tensor(
            [self.view_selection_outputs[f'{scene}_{view}'] for scene, view in zip(scene_names, target_views)], 
            device=batch['all_src_inps'].device
        ) # (B, K)
        
        depth, std, near_far = [None]*K, [None]*K, [None]*K
        feats = self.forward_feat(batch['all_src_inps'])
        
        selected_views = selected_views[k_best]
        volume_render_ret = {}
        for i in range(cfg.enerf.cas_config.num):
            mlp_level_ret = {}
            for k in range(K):
                # B, N, C, H, W -> B, I, C, H, W
                selected_views_indices = selected_views[:, k]
                batch_indices = torch.arange(B, device=batch['all_src_inps'].device).unsqueeze(-1).expand(-1, I)
                full_indices = torch.stack([batch_indices, selected_views_indices], dim=-1)
                batch['src_inps'] = batch['all_src_inps'][full_indices[:, :, 0], full_indices[:, :, 1]] # B, I, C, H, W
                batch['src_exts'] = batch['all_src_exts'][full_indices[:, :, 0], full_indices[:, :, 1]] # B, I, C, H, W
                batch['src_ixts'] = batch['all_src_ixts'][full_indices[:, :, 0], full_indices[:, :, 1]] # B, I, C, H, W
                
                feature_volume, depth_values, near_far[k] = utils.build_feature_volume(
                        feats[f'level_{i}'][full_indices[:, :, 0], full_indices[:, :, 1]], # B, I, feat_dim, H*s, W*s
                        batch,
                        D=cfg.enerf.cas_config.volume_planes[i],
                        depth=depth[k],
                        std=std[k],
                        near_far=near_far[k],
                        level=i)
                feature_volume, depth_prob = getattr(self, f'cost_reg_{i}')(feature_volume)
                depth[k], std[k] = utils.depth_regression(depth_prob, depth_values, i, batch)
                if not cfg.enerf.cas_config.render_if[i]:
                    continue
                rays = utils.build_rays(depth[k], std[k], batch, self.training, near_far[k], i)
                # UV(2) +  ray_o (3) + ray_d (3) + ray_near_far (2) + volume_near_far (2)
                im_feat_level = cfg.enerf.cas_config.render_im_feat_level[i]
                mlp_view_ret = self.batchify_rays_for_mlp(
                        rays=rays,
                        feature_volume=feature_volume,
                        batch=batch,
                        im_feat=feats[f'level_{im_feat_level}'][full_indices[:, :, 0], full_indices[:, :, 1]], # B, I, feat_dim, H*s, W*s
                        nerf_model=getattr(self, f'nerf_{i}'),
                        level=i)
                mlp_level_ret.update({key+f'_view{k}': mlp_view_ret[key] for key in mlp_view_ret})
            
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            volume_render_level_ret = self.merge_mlp_outputs(mlp_level_ret, K)
            
            if cfg.enerf.cas_config.depth_inv[i]:
                volume_render_level_ret.update({'depth_mvs': 1./depth[0]})
            else:
                volume_render_level_ret.update({'depth_mvs': depth[0]})
            volume_render_level_ret.update({'std': std[0]})
            if volume_render_level_ret['rgb'].isnan().any():
                __import__('ipdb').set_trace()
            volume_render_ret.update({key+f'_level{i}': volume_render_level_ret[key] for key in volume_render_level_ret})
            
        return volume_render_ret
