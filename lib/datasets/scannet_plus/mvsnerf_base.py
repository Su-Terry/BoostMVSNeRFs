import numpy as np
import os
import imageio
import cv2
import torch
from PIL import Image
from lib.config import cfg
from lib.datasets import enerf_utils

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w_ori = kwargs['input_h_w']
        im = Image.new('RGB', self.input_h_w_ori)
        im = np.array(im)
        im = im[32:-32, 32:-32]
        self.input_h_w = im.shape
        
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas()

    def build_metas(self):
        if len(self.scenes) == 0:
            # scenes = ['scene0000_01', 'scene0079_00', 
            #           'scene0158_00', 'scene0316_00',
            #           'scene0521_00', 'scene0553_00',
            #           'scene0616_00', 'scene0653_00']
            scenes = ['scene0000_01']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        
        def filter_valid_id(scene, id_list):
            empty_lst=[]
            for id in id_list:
                c2w = np.loadtxt(os.path.join(self.data_root, scene, "exported/pose", "{}.txt".format(id))).astype(np.float32)
                # filter nan, -inf, inf
                if np.max(np.abs(c2w)) < 30:
                    empty_lst.append(id)
            return empty_lst

        for scene in scenes:
            colordir = os.path.join(self.data_root, scene, "exported/color")
            image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
            image_paths = [os.path.join(self.data_root, scene, "exported/color/{}.jpg".format(i)) for i in range(len(image_paths))]
            pose_paths = [os.path.join(self.data_root, scene, "exported/pose/{}.txt".format(i)) for i in range(len(image_paths))]
            self.all_id_list = filter_valid_id(scene, list(range(len(image_paths))))
            
            poses = []
            for pose_file in pose_paths:
                pose = np.loadtxt(pose_file).astype(np.float32)
                poses.append(pose)
            poses = np.stack(poses)
            
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws = poses
            # c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            
            ixt_root = os.path.join(self.data_root, scene, 'exported', 'intrinsic')
            ixt_file = os.path.join(ixt_root, 'intrinsic_color.txt')
            ixt = np.loadtxt(ixt_file).astype(np.float32)[:3,:3]
            print(ixt[0, 2], ixt[1, 2])

            focal = [ixt[0, 0]* self.input_h_w[0]/ixt[0, 2], ixt[1, 1]* self.input_h_w[1]/ixt[1, 2]]
            directions = self.get_ray_directions(self.input_h_w[1], self.input_h_w[0], focal)
            ixt = np.array([[focal[0], 0, self.input_h_w[0]/2], [0, focal[1], self.input_h_w[1]/2], [0, 0, 1]])

            # ixt[:2, :2] *= 2
            ixts = np.tile(ixt, (len(c2ws), 1, 1))

            # Poses bounds
            depth_ranges = np.full((len(image_paths), 2), [0.25, 6]).astype(np.float32)
            
            image_names = [os.path.basename(image_path) for image_path in image_paths]
            
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': image_names, 'depth_ranges': depth_ranges.astype(np.float32)}
            scene_info['scene_name'] = scene
            self.scene_infos[scene] = scene_info
            
            train_file = f"data/scannet_plus/{scene}/train.txt"
            test_file = f"data/scannet_plus/{scene}/test.txt"
            
            if os.path.exists(train_file):
                train_ids = np.loadtxt(train_file, dtype="U")
                render_ids = np.loadtxt(test_file, dtype="U")
                train_ids = [int(os.path.basename(f).split('.')[0]) for f in train_ids]
                render_ids = [int(os.path.basename(f).split('.')[0]) for f in render_ids]
                if self.split == 'train':
                    render_ids = train_ids
            else:
                raise f"Train file, {train_file}, not exists."

            c2ws = c2ws[train_ids]
                        
            for i in render_ids:
                c2w = scene_info['c2ws'][i]
                ### 與render view uv重疊區域大小，depth
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                if self.split == 'train':
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.train_input_views[1]+1]]
                else:
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.test_input_views]]
                
                self.metas += [(scene, i, src_views, directions)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views, directions = self.metas[index]
                
        scene_info = self.scene_infos[scene]
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
                
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'all_src_inps': src_inps.transpose(0, 3, 1, 2),
               'all_src_exts': src_exts,
               'all_src_ixts': src_ixts})
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})
            

        H, W = tar_img.shape[:2]
        depth_ranges = np.array(scene_info['depth_ranges'])
        near_far = np.array([depth_ranges[:, 0].min().item(), depth_ranges[:, 1].max().item()]).astype(np.float32)
        # near_far = scene_info['depth_ranges'][tar_view]
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0, 'split': self.split}})
        ret.update({'depth_ranges': depth_ranges})

        for i in range(cfg.enerf.cas_config.num):
            _, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            
            rays_o, rays_d = self.get_rays(directions, torch.from_numpy(np.linalg.inv(tar_ext)[:3, :]))
            rays = torch.cat([rays_o, rays_d, 0.25*torch.ones_like(rays_o[:, :1]), 6*torch.ones_like(rays_o[:, :1])], 1)

            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            # s = cfg.enerf.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H), f'w_{i}': int(W)})
            
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
            ixt, ext, _ = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        # ixt[0] *= self.input_h_w[1] / orig_size[0]
        # ixt[1] *= self.input_h_w[0] / orig_size[1]
        return ixt, np.linalg.inv(ext), 1

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'exported', 'color', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        img = cv2.resize(img, self.input_h_w_ori[::-1], interpolation=cv2.INTER_AREA)
        img = np.array(img)
        img = img[32:-32, 32:-32]
        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K