import numpy as np
import os
from lib.config import cfg
import imageio
import cv2
from lib.config import cfg
from lib.datasets import enerf_utils

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
            scenes = ['scene0000_01', 'scene0079_00', 
                      'scene0158_00', 'scene0316_00',
                      'scene0521_00', 'scene0553_00',
                      'scene0616_00', 'scene0653_00']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        
        for scene in scenes:
            colordir = os.path.join(self.data_root, scene, "exported/color")
            image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f))]
            image_paths = [os.path.join(self.data_root, scene, "exported/color/{}.jpg".format(i)) for i in range(len(image_paths))]
            pose_paths = [os.path.join(self.data_root, scene, "exported/pose/{}.txt".format(i)) for i in range(len(image_paths))]
            
            poses = []
            for pose_file in pose_paths:
                pose = np.loadtxt(pose_file).astype(np.float32)
                poses.append(pose)
            poses = np.stack(poses)
            
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws = poses
            
            ixt_root = os.path.join(self.data_root, scene, 'exported', 'intrinsic')
            ixt_file = os.path.join(ixt_root, 'intrinsic_color.txt')
            ixt = np.loadtxt(ixt_file).astype(np.float32)[:3,:3]
            ixts = np.tile(ixt, (len(c2ws), 1, 1))
            # ixts[:, 0, 2], ixts[:, 1, 2] = ixts[:, 0, 2] / 2, ixts[:, 1, 2] / 2

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
                ### Take nearest views. (ENeRF)
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
                argsorts = distance.argsort()
                argsorts = argsorts[1:] if i in train_ids else argsorts
                if self.split == 'train':
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.train_input_views[1]]]
                else:
                    src_views = [train_ids[i] for i in argsorts[:cfg.enerf.test_input_views]]
                
                self.metas += [(scene, i, src_views)]
                
    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
                
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

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            if self.split == 'test':
                tmp_tar_img = cv2.resize(tar_img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
                tmp_tar_mask = cv2.resize(tar_mask, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
                rays, _, _ = enerf_utils.build_rays(tmp_tar_img, tar_ext, tar_ixt, tmp_tar_mask, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            # s = cfg.enerf.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H), f'w_{i}': int(W)})
        if self.split == 'test' and cfg.enerf.cas_config.num == 2:
            ret['rgb_0'], ret['msk_0'] = ret['rgb_1'], ret['msk_1']
            
        return ret

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
        if self.split == 'train':
            # Fixed image size
            img, orig_size = self.read_image(scene, view_idx, is_gt=False)
        else:
            # Original image size for evaluation
            img, orig_size = self.read_image(scene, view_idx, is_gt=True)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        ixt[0, 2] = self.input_h_w[1] / 2
        ixt[1, 2] = self.input_h_w[0] / 2
        return ixt, np.linalg.inv(ext), 1
    
    def read_image(self, scene, view_idx, is_gt=False):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'exported', 'color', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        if not is_gt:
            img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K