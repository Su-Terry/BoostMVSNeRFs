import numpy as np
import os
from glob import glob
from lib.utils.data_utils import load_K_Rt_from_P, read_cam_file
from lib.config import cfg
import imageio
import tqdm
from multiprocessing import Pool
import copy
import cv2
import random
from lib.config import cfg
from lib.utils import data_utils
from PIL import Image
import torch
from lib.datasets import enerf_utils
from torchvision import transforms as T

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['grass', 'hydrant', 'lab', 'pillar', 'road', 'sky', 'stair']
            # scenes = ['hydrant']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        # pairs = torch.load('data/mvsnerf/pairs.th')
        for scene in scenes:

            pose_bounds = np.load(os.path.join(self.data_root, scene, 'poses_bounds.npy')) # c2w, -u, r, -t
            poses = pose_bounds[:, :15].reshape((-1, 3, 5))

            h_, w_, self.focal = poses[0, :, -1]
            self.focal = [self.focal * self.input_h_w[1] / w_, self.focal * self.input_h_w[0] / h_]
            directions = self.get_ray_directions(self.input_h_w[0], self.input_h_w[1], self.focal)

            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            ixts = np.eye(3)[None].repeat(len(poses), 0)
            ixts[:, 0, 0], ixts[:, 1, 1] = poses[:, 2, 4], poses[:, 2, 4]
            ixts[:, 0, 2], ixts[:, 1, 2] = poses[:, 1, 4]/2., poses[:, 0, 4]/2.
            # self.poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
            # self.poses = self.poses @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            ixts[:, :2] *= 0.5

            # img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, scene, 'images_4')) if '.png' in item])
            img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, scene, 'images')) if '.png' in item or '.JPG' in item or '.jpg' in item])
            depth_ranges = pose_bounds[:, -2:]
            bounds = depth_ranges
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': img_paths, 'depth_ranges': depth_ranges.astype(np.float32)}
            scene_info['scene_name'] = scene
            self.scene_infos[scene] = scene_info

            all_ids = [i for i in range(len(img_paths))]
            train_ids = [i for i in all_ids if i % 8 != 0]
            if self.split == 'train':
                render_ids = train_ids
            else:
                render_ids = [i for i in all_ids if i % 8 == 0]

            c2ws = c2ws[train_ids]
            for i in render_ids:
                c2w = scene_info['c2ws'][i]
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                if self.split == 'train':
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.train_input_views[1]]]
                else:
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.test_input_views]]
                # print(argsorts)
                self.metas += [(scene, i, src_views, depth_ranges[src_views], directions)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views, depth_ranges, directions = self.metas[index]
        # if self.split == 'train':
        #     if np.random.random() < 0.1:
        #         src_views = src_views + [tar_view]
        #     src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]

        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        # print(src_inps.min(), src_inps.max(), tar_img.min(), tar_img.max())

        # src_inps = self.src_transform(torch.from_numpy(src_inps.transpose(0, 3, 1, 2)))

        ret = {'all_src_inps': src_inps.transpose(0, 3, 1, 2),
        # ret = {'all_src_inps': src_inps,
               'all_src_exts': src_exts,
               'all_src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})

        H, W = tar_img.shape[:2]
        # depth_ranges = np.array(scene_info['depth_ranges'])
        # print(depth_ranges)
        # print(src_views[:3])
        # near_far = np.array([depth_ranges[:3, 0].min()*0.8, depth_ranges[src_views[:3], 1].max()*1.2]).astype(np.float32)
        # near_far = scene_info['depth_ranges'][tar_view]
        # ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})
        ret.update({'depth_ranges': depth_ranges.astype(np.float32)})

        for i in range(cfg.enerf.cas_config.num):
            # print(tar_img.max(), tar_img.min())
            _, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            # print('c2w', tar_ext[:3, :])
            rays_o, rays_d = self.get_rays(directions, torch.from_numpy(np.linalg.inv(tar_ext)[:3, :]))
            rays = torch.cat([rays_o, rays_d, depth_ranges.min() * torch.ones_like(rays_o[:, :1]) * 0.8, depth_ranges.max() * torch.ones_like(rays_o[:, :1]) * 1.2], 1)
            # print("rays", tar_view, rays_o, rays_d)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            # s = cfg.enerf.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H), f'w_{i}': int(W)})
        # ret[f'rgb_0'] = ret[f'rgb_1']
        # ret[f'msk_0'] = ret[f'msk_1']
        return ret
    
    def get_rays(self, directions, c2w):
        rays_d = directions @ c2w[:3, :3].T
        rays_o = c2w[:3, 3].expand(rays_d.shape)

        rays_d = rays_d.view(-1, 3)
        rays_o = rays_o.view(-1, 3)
        return rays_o, rays_d
    
    def get_ray_directions(self, H, W, focal, center=None):
        grid = self.create_meshgrid(H, W)[0]
        i, j = grid.unbind(-1)
        cent = center if center is not None else [W/2., H/2.]  
        directions = torch.stack([(i-cent[0])/focal[0], (j-cent[1])/focal[1], torch.ones_like(i)], -1)
        return directions

    def create_meshgrid(self, height, width, normalized_coordinates=False):
        xs = torch.linspace(0, width - 1, width)
        ys = torch.linspace(0, height - 1, height)
        # Fix TracerWarning
        # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
        #       tensors will be generated.
        # Below is the code using normalize_pixel_coordinates:
        # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
        # if normalized_coordinates:
        #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
        # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
        if normalized_coordinates:
            xs = (xs / (width - 1) - 0.5) * 2
            ys = (ys / (height - 1) - 0.5) * 2
        # generate grid by stacking coordinates
        # TODO: torchscript doesn't like `torch_version_ge`
        # if torch_version_ge(1, 13, 0):
        #     x, y = torch_meshgrid([xs, ys], indexing="xy")
        #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
        # TODO: remove after we drop support of old versions
        # base_grid = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
        base_grid = torch.stack(torch.meshgrid([xs, ys]), dim=2)  # WxHx2
        return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2

    def read_src(self, scene, src_views):
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img, orig_size = self.read_image(scene, idx)
            imgs.append(((img/255.)*2-1).astype(np.float32))
            # imgs.append((img/255.).astype(np.float32))
            ixt, ext, _ = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx, False)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        return ixt, np.linalg.inv(ext), 1

    def read_image(self, scene, view_idx, is_gt=False):
        # image_path = os.path.join(self.data_root, scene['scene_name'], 'images_4', scene['image_names'][view_idx])
        image_path = os.path.join(self.data_root, scene['scene_name'], 'images_2', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        img = Image.fromarray(img.astype(np.uint8))
        if not is_gt:
            img = img.resize(self.input_h_w[::-1], Image.LANCZOS)

        # img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_LANCZOS4)
        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

