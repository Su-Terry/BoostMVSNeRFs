from lib.config import cfg
import numpy as np
import imageio
import os
import cv2
from PIL import Image

class Visualizer:
    def __init__(self,):
        self.write_video = cfg.write_video
        self.imgs = []
        self.depths = []
        self.imgs_coarse = []
        os.system('mkdir -p {}'.format(cfg.result_dir))
        os.system('mkdir -p {}'.format(cfg.result_dir + '/imgs'))

    def visualize(self, output, batch):
        B, S, _, H, W = batch['src_inps'].shape
        i = cfg.enerf.cas_config.num - 1
        render_scale = cfg.enerf.cas_config.render_scale[i]
        h_render, w_render = int(H*render_scale), int(W*render_scale)
        assert(B == 1)
        pred_rgb = output[f'rgb_level{i}'].reshape(h_render, w_render, 3).detach().cpu().numpy()
        pred_rgb = Image.fromarray((pred_rgb*255.).astype(np.uint8))
        pred_rgb = pred_rgb.resize((640, 480), Image.LANCZOS)
        pred_rgb = np.array(pred_rgb).astype(np.float32) / 255.
        depth = output[f'depth_level{i}'].reshape(h_render, w_render).detach().cpu().numpy()
        depth = Image.fromarray(depth)
        depth = depth.resize((640, 480), Image.LANCZOS)
        depth = np.array(depth)
        
        self.imgs.append(pred_rgb)
        self.depths.append(depth)
        if cfg.save_result:
            frame_id = batch['meta']['frame_id'][0].item()
            pred_rgb = (np.array(pred_rgb)*255).astype(np.uint8)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_rgb.jpg'.format(frame_id)), pred_rgb)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_dpt.jpg'.format(frame_id)), ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8))

    def summarize(self):
        img_rgb = (np.array(self.imgs)*255).astype(np.uint8)
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color.mp4'), img_rgb, fps=60)
        
        depths_rgb = []
        for depth in self.depths:
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = cv2.applyColorMap((depth*255).astype(np.uint8), cv2.COLORMAP_JET)
            depths_rgb.append(depth)
        
        imageio.mimwrite(os.path.join(cfg.result_dir, 'depth.mp4'), depths_rgb, fps=60)
        print('Save visualization results into {}'.format(cfg.result_dir))


