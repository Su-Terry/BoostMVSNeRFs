import matplotlib.pyplot as plt
from lib.utils import data_utils
from lib.utils import img_utils
from lib.config import cfg
import numpy as np
import torch.nn.functional as F
import torch
import imageio
import os
import cv2
from skimage.exposure import rescale_intensity

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
        h, w = int(H*render_scale), int(W*render_scale)
        assert(B == 1)
        pred_rgb = output[f'rgb_level{i}'].reshape(h, w, 3).detach().cpu().numpy()
        # depth = output[f'depth_level{i}'].reshape(h, w).detach().cpu().numpy()
        crop_h, crop_w = int(h * 0.1), int(w * 0.1)
        # pred_rgb = pred_rgb[crop_h:, crop_w:-crop_w]
        # depth = depth[crop_h:, crop_w:-crop_w]
        self.imgs.append(pred_rgb)
        # self.depths.append(depth)
        if cfg.save_result:
            frame_id = batch['meta']['frame_id'][0].item()
            pred_rgb = (np.array(pred_rgb)*255).astype(np.uint8)
            imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_rgb.jpg'.format(frame_id)), pred_rgb)
            # imageio.imwrite(os.path.join(cfg.result_dir, 'imgs/{:06d}_dpt.jpg'.format(frame_id)), ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8))

    def summarize(self):
        img_rgb = (np.array(self.imgs)*255).astype(np.uint8)
        imageio.mimwrite(os.path.join(cfg.result_dir, 'color.mp4'), img_rgb, fps=30)
        
        color1 = (0, 0, 255)     #red
        color2 = (0, 165, 255)   #orange
        color3 = (0, 255, 255)   #yellow
        color4 = (255, 255, 0)   #cyan
        color5 = (255, 0, 0)     #blue
        color6 = (128, 64, 64)   #violet
        colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)
        # resize lut to 256 (or more) values
        lut = cv2.resize(colorArr, (256,1), interpolation = cv2.INTER_LINEAR)
        
        # Color depth map
        # depths_gray = [ ((dpt - dpt.min()) / (dpt.max() - dpt.min()) * 255).astype(np.uint8) for dpt in self.depths ]
        # depths_gray = [ rescale_intensity(dpt, in_range='image', out_range=(0, 255)).astype(np.uint8) for dpt in depths_gray ]
        # depths_rgb = [ cv2.merge((dpt, dpt, dpt)) for dpt in depths_gray ]
        
        # apply lut
        # depths_rgb = [ cv2.LUT(dpt, lut) for dpt in depths_rgb ]
        
        # imageio.mimwrite(os.path.join(cfg.result_dir, 'depth.mp4'), depths_rgb, fps=30)
        print('Save visualization results into {}'.format(cfg.result_dir))


