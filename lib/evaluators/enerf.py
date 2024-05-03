import numpy as np
from lib.config import cfg
import os
import imageio
from lib.utils import img_utils
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch.nn.functional as F
import torch
import lpips
import imageio
from lib.utils import img_utils
import cv2
from PIL import Image


class Evaluator:

    def __init__(self,):
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        self.loss_fn_vgg = lpips.LPIPS(net='vgg')
        self.loss_fn_vgg.cuda()
        if cfg.enerf.eval_depth:
            # Following the setup of MVSNeRF
            self.eval_depth_scenes = ['scan1', 'scan8', 'scan21', 'scan103', 'scan110']
            self.abs = []
            self.acc_2 = []
            self.acc_10 = []
            self.mvs_abs = []
            self.mvs_acc_2 = []
            self.mvs_acc_10 = []
        os.system('mkdir -p ' + cfg.result_dir)

    def evaluate(self, output, batch):
        B, S, _, H, W = batch['src_inps'].shape
        for i in range(cfg.enerf.cas_config.num):
            if not cfg.enerf.cas_config.render_if[i]:
                continue
            H, W = batch['src_inps'].shape[-2:]
            s = cfg.enerf.cas_config.render_scale[i]
            H, W = int(H*s), int(W*s)
            if batch['meta']['split'] == 'train':
                h, w = H, W
            else:
                h, w = batch['meta'][f'h_{i}'], batch['meta'][f'w_{i}']
            
            pred_rgb_ims = []
            for b in range(B):
                pred_rgb_im = Image.fromarray((output[f'rgb_level{i}'][b].reshape(H, W, 3).detach().cpu().numpy()*255.).astype(np.uint8))
                # save the image
                img_path = f'tmp_{b}.png'
                pred_rgb_im.save(img_path)
                pred_rgb_im = pred_rgb_im.resize((w, h), Image.LANCZOS)
                pred_rgb_ims.append(np.array(pred_rgb_im).astype(np.float32) / 255.)
            
            pred_rgb = np.stack(pred_rgb_ims, axis=0)
            gt_rgb   = batch[f'rgb_{i}'].reshape(B, h, w, 3).detach().cpu().numpy()

            masks = (batch[f'msk_{i}'].reshape(B, h, w).cpu().numpy() >= 1).astype(np.uint8)

            if cfg.enerf.eval_center:
                H_crop, W_crop = int(h*0.1), int(w*0.1)
                pred_rgb = pred_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                gt_rgb = gt_rgb[:, H_crop:-H_crop, W_crop:-W_crop]
                masks = masks[:, H_crop:-H_crop, W_crop:-W_crop]

            for b in range(B):
                if not batch['meta']['scene'][b]+f'_level{i}' in self.scene_psnrs:
                    self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'] = []
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'] = []
                # self.scene_lpips[batch['meta']['scene'][b]] = []
                if cfg.save_result:
                    img = img_utils.horizon_concate(gt_rgb[b], pred_rgb[b])
                    img_path = os.path.join(cfg.result_dir, '{}_{}_{}.png'.format(batch['meta']['scene'][b], batch['meta']['tar_view'][b].item(), batch['meta']['frame_id'][b].item()))
                    imageio.imwrite(img_path, (img*255.).astype(np.uint8))

                mask = masks[b] == 1
                gt_rgb[b][mask==False] = 0.
                pred_rgb[b][mask==False] = 0.

                psnr_item = psnr(gt_rgb[b][mask], pred_rgb[b][mask], data_range=1.)
                if i == cfg.enerf.cas_config.num-1:
                    self.psnrs.append(psnr_item)
                self.scene_psnrs[batch['meta']['scene'][b]+f'_level{i}'].append(psnr_item)

                ssim_item = ssim((gt_rgb[b]*255).astype(np.uint8), (pred_rgb[b]*255).astype(np.uint8), channel_axis=-1)
                if i == cfg.enerf.cas_config.num-1:
                   self.ssims.append(ssim_item)
                self.scene_ssims[batch['meta']['scene'][b]+f'_level{i}'].append(ssim_item)

                if cfg.eval_lpips:
                    gt, pred = torch.Tensor(gt_rgb[b])[None].permute(0, 3, 1, 2), torch.Tensor(pred_rgb[b])[None].permute(0, 3, 1, 2)
                    gt, pred = (gt-0.5)*2., (pred-0.5)*2.
                    lpips_item = self.loss_fn_vgg(gt.cuda(), pred.cuda()).item()
                    if i == cfg.enerf.cas_config.num-1:
                        self.lpips.append(lpips_item)
                    self.scene_lpips[batch['meta']['scene'][b]+f'_level{i}'].append(lpips_item)

                if cfg.enerf.eval_depth and (i == cfg.enerf.cas_config.num - 1) and batch['meta']['scene'][b] in self.eval_depth_scenes:
                    nerf_depth = output['depth_level1'].cpu().numpy()[b].reshape((h, w))
                    mvs_depth = output['depth_mvs_level1'].cpu().numpy()[b]
                    nerf_gt_depth = batch['tar_dpt'][b].cpu().numpy().reshape(*nerf_depth.shape)
                    mvs_gt_depth = cv2.resize(nerf_gt_depth, mvs_depth.shape[::-1], interpolation=cv2.INTER_NEAREST)
                    # nerf_mask = np.logical_and(nerf_gt_depth > 425., nerf_gt_depth < 905.)
                    # mvs_mask = np.logical_and(mvs_gt_depth > 425., mvs_gt_depth < 905.)
                    nerf_mask = nerf_gt_depth != 0.
                    mvs_mask = mvs_gt_depth != 0.
                    self.abs.append(np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]).mean())
                    self.acc_2.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 2).mean())
                    self.acc_10.append((np.abs(nerf_depth[nerf_mask] - nerf_gt_depth[nerf_mask]) < 10).mean())
                    self.mvs_abs.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask])).mean())
                    self.mvs_acc_2.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 2.).mean())
                    self.mvs_acc_10.append((np.abs(mvs_depth[mvs_mask] - mvs_gt_depth[mvs_mask]) < 10.).mean())

    def summarize(self):
        ret = {}
        ret.update({'psnr': np.mean(self.psnrs)})
        ret.update({'ssim': np.mean(self.ssims)})
        if cfg.eval_lpips:
            ret.update({'lpips': np.mean(self.lpips)})
        print('='*30)
        for scene in self.scene_psnrs:
            if cfg.eval_lpips:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} lpips:{:.3f}'.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene]), np.mean(self.scene_lpips[scene])))
            else:
                print(scene.ljust(16), 'psnr: {:.2f} ssim: {:.3f} '.format(np.mean(self.scene_psnrs[scene]), np.mean(self.scene_ssims[scene])))
        print('='*30)
        print(ret)
        if cfg.enerf.eval_depth:
            depth_ret = {}
            keys = ['abs', 'acc_2', 'acc_10']
            for key in keys:
                depth_ret[key] = np.mean(getattr(self, key))
                setattr(self, key, [])
            print(depth_ret)
            keys = ['mvs_abs', 'mvs_acc_2', 'mvs_acc_10']
            depth_ret = {}
            for key in keys:
                depth_ret[key] = np.mean(getattr(self, key))
                setattr(self, key, [])
            print(depth_ret)
        self.psnrs = []
        self.ssims = []
        self.lpips = []
        self.scene_psnrs = {}
        self.scene_ssims = {}
        self.scene_lpips = {}
        if cfg.save_result:
            print('Save visualization results to: {}'.format(cfg.result_dir))
        return ret
