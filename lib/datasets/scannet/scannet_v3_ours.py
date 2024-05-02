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
import os, shutil, imageio
import png

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
        
        def get_key(item):
            return int(os.path.basename(item).split('.')[0])
        
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
            depth_image_paths = [os.path.join(self.data_root, scene, "exported/depth/{}.png".format(i)) for i in range(len(image_paths))]
            pose_paths = [os.path.join(self.data_root, scene, "exported/pose/{}.txt".format(i)) for i in range(len(image_paths))]
            self.all_id_list = filter_valid_id(scene, list(range(len(image_paths))))
            # print(all_id_list)
            
            poses = []
            for pose_file in pose_paths:
                with open(pose_file, 'r') as f:
                    lines = f.readlines()
                    pose = np.loadtxt(pose_file).astype(np.float32)
                    poses.append(pose)
            poses = np.stack(poses)
            
            c2ws = np.eye(4)[None].repeat(len(poses), 0)
            c2ws = poses
            # c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
            
            ixt_root = os.path.join(self.data_root, scene, 'exported', 'intrinsic')
            ixt_file = os.path.join(ixt_root, 'intrinsic_color.txt')
            ixt = np.loadtxt(ixt_file).astype(np.float32)[:3,:3]
            ixts = np.tile(ixt, (len(c2ws), 1, 1))
            
            depth_ixt_file = os.path.join(ixt_root, 'intrinsic_depth.txt')
            depth_ixt = np.loadtxt(depth_ixt_file).astype(np.float32)[:3,:3]
            depth_ixts = np.tile(depth_ixt, (len(c2ws), 1, 1))
            # ixts[:, 0, 2], ixts[:, 1, 2] = ixts[:, 0, 2] / 2, ixts[:, 1, 2] / 2

            # depth_ranges = pose_bounds[:, -2:]
            depth_ranges = np.ones((len(image_paths), 2))
            depth_ranges[:, 0] *= 0.25
            depth_ranges[:, 1] *= 6
            
            image_names = [os.path.basename(image_path) for image_path in image_paths]
            depth_image_names = [os.path.basename(depth_image_path) for depth_image_path in depth_image_paths]
            
            scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': image_names, 'depth_ranges': depth_ranges.astype(np.float32),
                          'depth_ixts': depth_ixts.astype(np.float32), 'depth_image_names': depth_image_names}
            scene_info['scene_name'] = scene
            self.scene_infos[scene] = scene_info
            
            train_file = f"data/scannet/{scene}/train.txt"
            test_file = f"data/scannet/{scene}/test.txt"
            
            if os.path.exists(train_file):
                train_ids = np.loadtxt(train_file, dtype="U")
                render_ids = np.loadtxt(test_file, dtype="U")
                train_ids = [int(os.path.basename(f).split('.')[0]) for f in train_ids]
                render_ids = [int(os.path.basename(f).split('.')[0]) for f in render_ids]
                if self.split == 'train':
                    render_ids = train_ids
            else:
                print("train.txt not exist")
                exit(1)
            
            # self.all_id_list = self.all_id_list[::4]
            # step = 8
            # render_ids = self.all_id_list[::step]
            # train_ids = [self.all_id_list[i] for i in range(len(self.all_id_list)) if (i % step) !=0]
            # if self.split == 'train':
            #     render_ids = train_ids
                
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
                
                self.metas += [(scene, i, src_views)]
                
            
            # save image to ../ENeRF_workspace/scannet_v2/{scene}/exported/color
            # save pose to ../ENeRF_workspace/scannet_v2/{scene}/exported/pose
            # save intrinsic to ../ENeRF_workspace/scannet_v2/{scene}/exported/intrinsic
            
            # cfg.workspace = '../ENeRF_workspace'
            # if not os.path.exists(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported')):
            #     os.makedirs(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported'))
            # if not os.path.exists(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'color')):
            #     os.makedirs(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'color'))
            # if not os.path.exists(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'depth')):
            #     os.makedirs(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'depth'))
            # if not os.path.exists(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'pose')):
            #     os.makedirs(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'pose'))
            # if not os.path.exists(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'intrinsic')):
            #     os.makedirs(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'intrinsic'))
            # for i in range(len(image_paths)):
                # rgb = self.read_image(self.scene_infos[scene], i)[0]
                # pose, ixt = self.read_pose_ixt(self.scene_infos[scene], i, self.read_image(self.scene_infos[scene], i)[1])
                # depth_rgb = self.read_depth(self.scene_infos[scene], i)[0]
                # depth_ixt = self.read_depth_ixt(self.scene_infos[scene], i, self.read_image(self.scene_infos[scene], i)[1])
                # rgb = rgb.astype(np.uint8)
                # imageio.imwrite(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'color', '{}.jpg'.format(i)), rgb)
                # np.savetxt(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'pose', '{}.txt'.format(i)), pose)
                # np.savetxt(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'intrinsic', 'intrinsic_color.txt'), ixt)
                # imageio.imwrite(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'depth', '{}.png'.format(i)), (depth_rgb.astype(np.float32)/65536*255).astype(np.uint8))
                # with open(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'depth', '{}.png'.format(i)), 'wb') as f:
                #     writer = png.Writer(width=depth_rgb.shape[1], height=depth_rgb.shape[0], bitdepth=16, greyscale=True)
                #     writer.write(f, depth_rgb.astype(np.uint16))
                
                # np.savetxt(os.path.join(cfg.workspace, 'scannet_v2', scene, 'exported', 'intrinsic', 'intrinsic_depth.txt'), depth_ixt)

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
                
        scene_info = self.scene_infos[scene]
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        
        ret = {'all_src_inps': src_inps.transpose(0, 3, 1, 2),
               'all_src_exts': src_exts,
               'all_src_ixts': src_ixts}
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
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.enerf.cas_config.num):
            rays, rgb, msk = enerf_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            s = cfg.enerf.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H*s), f'w_{i}': int(W*s)})
            
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
        img, orig_size = self.read_image(scene, view_idx)
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
    
    def read_pose_ixt(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        return ext, ixt
    
    def read_depth_ixt(self, scene, view_idx, orig_size):
        ixt = scene['depth_ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        return ixt

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'exported', 'color', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        img = img[32:-32, 32:-32]
        orig_size = img.shape[:2][::-1]
        img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        img = np.array(img)
        return np.array(img), orig_size
    
    def read_depth(self, scene, view_idx):
        image_path = os.path.join(self.data_root, scene['scene_name'], 'exported', 'depth', scene['depth_image_names'][view_idx])
        # read 16-bit png as 16-bit numpy array
        # depth = np.array(imageio.imread(image_path, format='PNG-FI')).astype(np.uint16)
        depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        depth = depth[32:-32, 32:-32]
        orig_size = depth.shape[:2][::-1]
        depth = cv2.resize(depth, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        depth = np.array(depth)
        return np.array(depth), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K