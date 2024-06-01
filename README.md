# BoostMVSNeRFs

[**BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes**](https://arxiv.org/abs/24XX.XXXXX)  
*Chih-Hai Su**, *Chih-Yao Hu**, *Shr-Ruei Tsai**, *Jie-Ying Lee**, *Chin-Yang Lin*, and *Yu-Lun Liu*  
*Proceedings of SIGGRAPH 2024*  
[Project Page](https://su-terry.github.io/BoostMVSNeRFs/) / Arxiv / Video / [Evaluation Images (2 GB)](https://drive.google.com/drive/folders/1u8njbeysuBgLihxmpGRY5vePm1aacfPi)


## News
- **06/01/2024** Code Release!

## Installation

### Set up the python environment
```bash
conda create -n boostmvsnerfs python=3.8
conda activate boostmvsnerfs
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# Note: Make sure CUDA is available and compatible with your system before installing these requirements (inplace-abn).
pip install -r requirements.txt
```

### Set up datasets

#### 0. Set up workspace and pre-trained models
The workspace is the disk directory that stores datasets, training logs, checkpoints and results. Please ensure it has enough space. 
```bash
export workspace=$PATH_TO_YOUR_WORKSPACE
```

The workspace should follow the following format.
```
workspace
    ├── free_dataset
    ├── scannet_plus
    └── trained_model
        ├── pretrain
        │   ├── enerf
        │   │   └── latest.pth        # ENeRF_pretrained model
        │   ├── enerf_ours
        │   │   └── latest.pth        # ENeRF_pretrained model
        │   ├── mvsnerf
        │   │   └── latest.pth        # MVSNeRF_pretrained model
        │   └── mvsnerf_ours
        │       └── latest.pth        # MVSNeRF_pretrained model
        ├── enerf_ft
        │   └── grass_ft              
        │       └── latest.pth        # ENeRF_pretrained model
        ├── enerf_ours_ft
        │   └── grass_ft              
        │       └── latest.pth        # ENeRF_pretrained model
        ├── mvsnerf_ft
        │   └── grass_ft              
        │       └── latest.pth        # MVSNeRF_pretrained model
        └── mvsnerf_ours_ft
            └── grass_ft              
                └── latest.pth        # MVSNeRF_pretrained model
```
Download the ENeRF pretrained model from [dtu_pretrain](https://drive.google.com/drive/folders/10vGC0_DuwLJwfy9OwUHhK7pRPoNP5rux?usp=share_link)  (pretrained on the DTU dataset).
Download the MVSNeRF pretrained model that fits our model backbone from [dtu_pretrain_mvs](https://drive.google.com/file/d/13OAVlcXgt7cGpFSTDvsy3SECH3KQEJ5N/view?usp=sharing) (pretrained on the DTU dataset).

<!-- #### 1. DTU
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip. [MVSNeRF](https://github.com/apchenstu/mvsnerf) provide a [DTU example](https://1drv.ms/u/s!AjyDwSVHuwr8zhAAXh7x5We9czKj?e=oStQ48), please follow with the example's folder structure.

```bash
mv dtu_example.zip $workspace
cd $workspace
unzip dtu_example.zip
```
-->

#### 1. Free Dataset
Download the [Free dataset](https://www.dropbox.com/sh/jmfao2c4dp9usji/AAC7Ydj6rrrhy1-VvlAVjyE_a?dl=0) from the original [F2-NeRF repo](https://github.com/Totoro97/f2-nerf).

#### 2. ScanNet_plus Dataset (originated from ScanNet)
We adjusted the original dataset by removing black areas from the images due to distortion in the ScanNet dataset. 
1. Before download ScanNet, please make sure you fill in this form before downloading the data: https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
2. Download the [ScanNet dataset](https://drive.google.com/file/d/1uu_xGSBI_gkepaN4FnAZs9lkD0eKffnu/view?usp=sharing).
<!-- 
Follow the guidelines to download the ScanNet dataset and convert it to our ScanNet_plus dataset. Big thanks to Point-NeRF for their installation guidelines, [link](https://github.com/Xharlie/pointnerf?tab=readme-ov-file#scannet).

1. Before download ScanNet, please make sure you fill in this form before downloading the data: https://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf
2. Download specific scenes (used by NSVF):
    ```
    python scripts/data/download-scannet.py -o $workspace/scannet_plus/ id scene0101_04
    python scripts/data/download-scannet.py -o $workspace/scannet_plus/ id scene0241_01
    ```
3. Process the sens files:
    ```
    python ScanNet/SensReader/python/reader.py --filename data_src/nrData/scannet/scans/scene0101_04/scene0101_04.sens  --output_path data_src/nrData/scannet/scans/scene0101_04/exported/ --export_depth_images --export_color_images --export_poses --export_intrinsics
    python ScanNet/SensReader/python/reader.py --filename data_src/nrData/scannet/scans/scene0241_01/scene0241_01.sens  --output_path data_src/nrData/scannet/scans/scene0241_01/exported/ --export_depth_images --export_color_images --export_poses --export_intrinsics
    ```
 -->
## Evaluation

#### 1. Evaluate the pre-trained model on the Free datasets.

```bash
bash scripts/exps/evaluate/free_pretrained.sh enerf
bash scripts/exps/evaluate/free_pretrained.sh enerf_ours
bash scripts/exps/evaluate/free_pretrained.sh mvsnerf
bash scripts/exps/evaluate/free_pretrained.sh mvsnerf_ours
```

#### 2. Evaluate the pre-trained model on the ScanNet_plus datasets.

```bash
bash scripts/exps/evaluate/scannet_plus_pretrained.sh enerf
bash scripts/exps/evaluate/scannet_plus_pretrained.sh enerf_ours
bash scripts/exps/evaluate/scannet_plus_pretrained.sh mvsnerf
bash scripts/exps/evaluate/scannet_plus_pretrained.sh mvsnerf_ours
```

## Fine-tuning

#### 1. On the grass scene of the Free dataset with ENeRF/ENeRF_ours/MVSNeRF/MVSNeRF_ours.
```bash
bash scripts/exps/finetune/free_ft.sh enerf        grass
bash scripts/exps/finetune/free_ft.sh enerf_ours   grass
bash scripts/exps/finetune/free_ft.sh mvsnerf      grass
bash scripts/exps/finetune/free_ft.sh mvsnerf_ours grass
```
#### 2. Evaluate the finetune results
```bash
bash scripts/exps/evaluate/free_ft.sh enerf        grass
bash scripts/exps/evaluate/free_ft.sh enerf_ours   grass
bash scripts/exps/evaluate/free_ft.sh mvsnerf      grass
bash scripts/exps/evaluate/free_ft.sh mvsnerf_ours grass
```

<!-- Fine-tuning for 11000 iterations takes about 90 minutes, on our test machine ( ? CPU, RTX 4090 GPU). -->


## Citation

If you find this code useful for your research, please use the following BibTeX entry.
