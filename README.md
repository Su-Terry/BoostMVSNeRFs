# BoostMVSNeRFs

[**BoostMVSNeRFs: Boosting MVS-based NeRFs to Generalizable View Synthesis in Large-scale Scenes**](https://arxiv.org/abs/24XX.XXXXX)  
*Chih-Hai Su**, *Chih-Yao Hu**, *Shr-Ruei Tsai**, *Jie-Ying Lee**, *Chin-Yang Lin*, and *Yu-Lun Liu*  
*Proceedings of SIGGRAPH 2024*  
[Project Page](https://su-terry.github.io/BoostMVSNeRFs/) / [Arxiv](https://arxiv.org/abs/24XX.XXXXX) / [Video]() / [Evaluation Images (X GB)]()


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

#### 0. Set up workspace
The workspace is the disk directory that stores datasets, training logs, checkpoints and results. Please ensure it has enough space. 
```bash
export workspace=$PATH_TO_YOUR_WORKSPACE
```
   
#### 1. Pre-trained model

Download the pretrained model from [dtu_pretrain](https://drive.google.com/drive/folders/10vGC0_DuwLJwfy9OwUHhK7pRPoNP5rux?usp=share_link) (Pretrained on DTU dataset.)

Put it into `$workspace/trained_model/pretrain/enerf/dtu_pretrain/latest.pth`.

#### 2. DTU
Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
and [Depth_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) from original [MVSNet repo](https://github.com/YoYo000/MVSNet)
and unzip. [MVSNeRF](https://github.com/apchenstu/mvsnerf) provide a [DTU example](https://1drv.ms/u/s!AjyDwSVHuwr8zhAAXh7x5We9czKj?e=oStQ48), please follow with the example's folder structure.

```bash
mv dtu_example.zip $workspace
cd $workspace
unzip dtu_example.zip
```

#### 2. Free Dataset

#### 3. ScanNet Dataset


## Evaluation

### Evaluate the pretrained model on DTU, Free and ScanNet datasets

```bash
python run.py --type evaluate --cfg_file configs/exps/evaluate/enerf/dtu_eval.yaml
python run.py --type evaluate --cfg_file configs/exps/evaluate/enerf/free_eval.yaml
python run.py --type evaluate --cfg_file configs/exps/evaluate/enerf/scannet_eval.yaml
```

## Training and fine-tuning

### Training
Use the following command to train a generalizable model on DTU.
```bash
python train_net.py --cfg_file configs/pretrain/enerf/dtu_pretrain.yaml 
```


### Fine-tuning on grass scene of Free dataset.

```bash
python train_net.py --cfg_file configs/exps/finetune/enerf/free/grass_ft.yaml
```

<!-- Fine-tuning for 3000 and 11000 iterations takes about 11 minutes and 40 minutes, respectively, on our test machine ( i9-12900K CPU, RTX 3090 GPU). -->


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

