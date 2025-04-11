# 版本号要求

Ubuntu 22.04 

CUDA 11.8 

确保输出中显示的驱动版本与 CUDA 11.8 兼容（建议驱动版本至少为 515 或更高）。

## 安装基础依赖：

apt-get update

apt-get install -y git wget nano unzip ffmpeg libsm6 libxext6

 

# 云服务器可以从这里开始：

### 创建和配置 Conda 环境：

conda create -n env python=3.10 -y

conda activate env

### 安装 PyTorch 和 CUDA

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

\# 验证

python 

import torch 

print(torch.cuda.is_available())

exit()

### 安装其他必要的 Conda 包  

 conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y

wget https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -O /tmp/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2
conda install /tmp/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -y

### 安装 Python 包  

pip install h5py hydra-core open_clip_torch supervision loguru
pip install wget iopath torchpack pyyaml tqdm
pip install opencv-python natsort imageio onnxruntime
pip install open3d==0.16.0 fast-pytorch-kmeans
conda install -c rapidsai -c nvidia -c conda-forge cuml=24.8 -y
pip install gdown
pip install xformers==0.0.22 
pip install imageio[ffmpeg]

### 下载和解压预训练模型

gdown 1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE -O ~/weights/mobilesamv2/weight.zip
unzip ~/weights/mobilesamv2/weight.zip -d ~/weights/mobilesamv2/
cp ~/MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt ~/weights/mobilesamv2/weight/

或者
手动从https://drive.usercontent.google.com/download?id=1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE&export=download&authuser=0
下载weight.zip到本地，解压缩。
将MobileSAM/MobileSAMv2/PromptGuidedDecoder/Prompt_guided_Mask_Decoder.pt复制到weight文件夹中

## 克隆并安装特定的 Git 仓库 

### 安装chamferdist

git clone https://github.com/krrish94/chamferdist.git
cd chamferdist
sed -i 's/c++14/c++17/' setup.py
pip install .
cd ..

### 安装gradslam

git clone https://github.com/gradslam/gradslam.git
cd gradslam
git checkout conceptfusion
git checkout 59ca872e3d265ad09f63c4793d011fad67064452
pip install .
cd ..

### 安装 Grounded-Segment-Anything

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything
cd Grounded-Segment-Anything/segment_anything
pip install -e .
cd ../..

### 安装MobileSAM

git clone https://github.com/ChaoningZhang/MobileSAM
cd MobileSAM
git checkout c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed
sed -i 's/"onnx",\s*//g; s/"onnxruntime",\s*//g' setup.py
sed -i 's/from \.export import \*//' MobileSAMv2/efficientvit/apps/utils/__init__.py
pip install -e .
cd ..

### 克隆和安装 LLaVA

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
git checkout c121f0432da27facab705978f83c4ada465e46fd
pip install --upgrade timm==0.6.13
sed -i 's/"torch==2\.1\.2",\s*"torchvision==0\.16\.2",\s*//g' pyproject.toml
pip install -e .

### 设置环境变量

vim ~/.bashrc
在文件末尾添加：
export PYTHONPATH=/data/coding/MobileSAM/MobileSAMv2:$PYTHONPATH
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1
source ~/.bashrc

### 检查环境

python
import torch
import pytorch3d
import open3d
print(torch.cuda.is_available())

## 下载模型

### LLAVA模型下载：

网址：https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b/tree/main

### CLIP模型下载：

网址：https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main

### 下载dinov2模型：（可以不用）

下载网址：https://github.com/facebookresearch/dinov2/zipball/main

下载dinov2权重：

下载网址：

https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth

https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth
