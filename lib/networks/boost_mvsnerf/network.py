from lib.config import cfg
from lib.networks.enerf.utils import *
import os
import torch
torch.autograd.set_detect_anomaly(True)
from lib.networks.mvsnerf import network
from lib.networks.mvsnerf.utils import get_ndc_coordinate
import json


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

        ray_tar = batch['rays_0'][0]
        world_xyz, z_vals = self.ray_marcher(ray_tar, 128)
        N_samples = 128

        B, S, _, H_O, W_O = batch['src_inps'].shape # B, S, C, H, W
        H, W = int(H_O * cfg.enerf.cas_config.render_scale[0]), int(W_O * cfg.enerf.cas_config.render_scale[0])
        inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=world_xyz.device)

        mask = mask_viewport(world_xyz, batch['src_exts'], batch['src_ixts'], inv_scale)
        mask = mask.reshape(B, -1, N_samples, 1) / N_samples
        mask = mask.repeat(1, 1, 1, 4) # rgb + alpha
        
        mask = raw2outputs(mask, z_vals, cfg.enerf.white_bkgd)['rgb'].mean(-1)
        mask_all = {
            f'mask_level0': mask
        }

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
        # world_xyz, uvd, z_vals = sample_along_depth(rays, N_samples=cfg.enerf.cas_config.num_samples[level], level=level)

        world_xyz, z_vals = self.ray_marcher(rays[0], cfg.enerf.cas_config.num_samples[level])

        B, N_rays, N_samples = world_xyz.shape[:3]
        chunk = N_rays // 10
        net_output = torch.zeros(B, N_rays, N_samples, 4, device=world_xyz.device)
        mask = torch.zeros(B, N_rays, N_samples, device=world_xyz.device)

        for i in range(0, N_rays, chunk):
            world_xyz_i = world_xyz[:, i:i + chunk]
            rays_i = rays[:, i:i + chunk]
            H_O, W_O = kwargs['batch']['src_inps'].shape[-2:]
            B, H, W = 1, int(H_O * cfg.enerf.cas_config.render_scale[level]), int(W_O * cfg.enerf.cas_config.render_scale[level])

            inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device)

            uvd_i = get_ndc_coordinate(batch['src_exts'][0][0], batch['src_ixts'][0][0], world_xyz_i[0], inv_scale, near=batch['near_far'].min(), far=batch['near_far'].max(), pad=24)[None]
            raw = self.rendering(batch, world_xyz_i[0], uvd_i, z_vals, rays_i[..., :3], rays_i[..., 3:6], feat_volume)

            net_output_i = nerf_model(raw)
            net_output_i = net_output_i.reshape(B, -1, N_samples, net_output_i.shape[-1])
        
            with torch.no_grad():
                inv_scale = torch.tensor([W-1, H-1], dtype=torch.float32, device=net_output.device)
                mask_i = mask_viewport(world_xyz_i, kwargs['batch']['src_exts'], kwargs['batch']['src_ixts'], inv_scale)
                mask_i = mask_i.reshape(B, -1, N_samples)

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

    def merge_mlp_outputs(self, outputs, K):
        net_outputs = torch.stack([outputs[f'net_output_view{i}'] for i in range(K)], dim=1)
        masks = torch.stack([outputs[f'mask_view{i}'] for i in range(K)], dim=1)
        z_vals = torch.stack([outputs[f'z_vals_view{i}'] for i in range(K)], dim=1)
        
        masks_sum = masks.unsqueeze(1).sum(2)
        masks = torch.where(masks_sum > 0, masks / masks_sum, 1 / K)
                    
        volume_render_outputs = raw2outputs_blend(net_outputs, masks, z_vals, cfg.enerf.white_bkgd)
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
        
        last = cfg.enerf.cas_config.num - 1
        D = cfg.enerf.cas_config.num_samples[last-1]

        feats = self.feature(batch['all_src_inps'][0]) # (B*V, C, H, W)
        feats = feats.view(1, -1, *feats.shape[1:])  # (B, V, C, h, w)
        
        selected_views = selected_views[k_best]
        mlp_level_ret = {}
        for k in range(K):
            selected_views_indices = selected_views[:, k]
            batch_indices = torch.arange(B, device=batch['all_src_inps'].device).unsqueeze(-1).expand(-1, I)
            full_indices = torch.stack([batch_indices, selected_views_indices], dim=-1)
                
            t_vals = torch.linspace(0., 1., steps=D, device=batch['all_src_inps'].device, dtype=batch['all_src_inps'].dtype)  # (B, D)
            near, far = (batch['depth_ranges'][full_indices[:, :, 0], full_indices[:, :, 1]].min()*0.8).to(batch['all_src_inps'].device), (batch['depth_ranges'][full_indices[:, :, 0], full_indices[:, :, 1]].max()*1.2).to(batch['all_src_inps'].device)
            depth_values = near * (1.-t_vals) + far * (t_vals)
            depth_values = depth_values.unsqueeze(0).to(batch['all_src_inps'].device)

            batch['near_far'] = torch.stack([near, far]).to(batch['all_src_inps'].device)
            batch['src_inps'] = batch['all_src_inps'][full_indices[:, :, 0], full_indices[:, :, 1]]
            batch['src_exts'] = batch['all_src_exts'][full_indices[:, :, 0], full_indices[:, :, 1]]
            batch['src_ixts'] = batch['all_src_ixts'][full_indices[:, :, 0], full_indices[:, :, 1]]
            proj_mats = self.get_proj_mats(batch).to(batch['all_src_inps'].device)
            volume_feat = self.build_volume_costvar_img(batch['src_inps'], feats[full_indices[:, :, 0], full_indices[:, :, 1]], proj_mats, depth_values, pad=24)
            volume_feat = self.cost_reg_2(volume_feat) # (B, 1, D, h, w)
            volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

            # build rays
            rays = batch['rays_0']

            # batchify rays for mlp
            mlp_view_ret = self.batchify_rays_for_mlp(rays, level=0, batch=batch, im_feat=feats[full_indices[:, :, 0], full_indices[:, :, 1]], feature_volume=volume_feat, nerf_model=self.nerf)

            mlp_level_ret.update({key+f'_view{k}': mlp_view_ret[key] for key in mlp_view_ret})

        volume_rendered_ret = self.merge_mlp_outputs(mlp_level_ret, K)
        volume_render_ret = {}
        volume_render_ret.update({key+f'_level0': volume_rendered_ret[key] for key in volume_rendered_ret})


        return volume_render_ret