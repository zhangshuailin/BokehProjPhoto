# 🎯 MODNet 人像Matting完整使用指南

## 📝 项目介绍

MODNet是一个**实时人像抠图模型**，可以从RGB图像直接生成Alpha Matte（透明度图）。无需trimap输入，直接输出高质量的人像蒙版。

预训练模型：`modnet_photographic_portrait_matting.ckpt`

**特点**：
- ✅ MODNet模型代码（`src/models/modnet.py`）
- ✅ 预训练权重文件（`pretrained/modnet_photographic_portrait_matting.ckpt`）
- ✅ 完整的推理脚本

---

## 🚀 5分钟快速上手

### 步骤1: 环境检查（必做）
```bash
# 首先检查环境和依赖
python check_environment.py
```

这个脚本会检查：
- ✅ Python版本 >= 3.6
- ✅ PyTorch安装
- ✅ CUDA可用性
- ✅ 依赖包（pillow, numpy等）
- ✅ MODNet代码文件
- ✅ 模型文件存在
- ✅ 模型能否正常加载

### 步骤2: 单张图片推理（推荐先试这个）

**最简单的用法**：
```bash
python simple_inference.py "你的图片.jpg"
```

**指定输出路径**：
```bash
python simple_inference.py "你的图片.jpg" "输出文件夹/输出.png"

# 使用CPU（如果没有GPU）
python simple_inference.py "你的图片.jpg" --device cpu
```

**Windows示例**：
```bash
python simple_inference.py "C:\Users\你的用户名\Desktop\portrait.jpg" "output.png"
```

### 步骤3: 批量处理多张图片
```bash
# 创建输入和输出文件夹
mkdir test_images output_mattes

# 复制你的图片到 test_images 文件夹

# 运行批量推理
python run_portrait_matting.py --input-path test_images --output-path output_mattes
```

---

## ⚙️ 环境配置

### 1. 检查Python版本
```bash
python --version  # 须 >= Python 3.6
```

### 2. 安装依赖
```bash
pip install torch torchvision pillow numpy
```

> 💡 如果需要GPU加速，请先安装对应的CUDA版本PyTorch：
> ```bash
> # CUDA 11.8版本
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> 
> # CUDA 12.1版本
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 📊 什么是Alpha Matte?

Alpha Matte是图像的**透明度图**（灰度图）：
- **黑色 (0)** = 完全透明（背景）
- **白色 (255)** = 完全不透明（人物）
- **灰色 (128)** = 半透明（头发等边界）

### 使用Alpha Matte的示例：

**1. 替换背景**
```python
from PIL import Image
import numpy as np

# 读取原图和matte
image = Image.open('portrait.jpg')
matte = Image.open('portrait_matte.png')

# 创建RGBA图像
rgba = image.convert('RGBA')
rgba.putalpha(matte)
rgba.save('portrait_with_alpha.png')
```

**2. 模糊背景**
```python
from PIL import Image, ImageFilter
import numpy as np

image = Image.open('portrait.jpg')
matte_arr = np.array(Image.open('portrait_matte.png')) / 255.0

# 模糊原图背景
blurred = image.filter(ImageFilter.GaussianBlur(radius=15))

# 按matte混合
for i in range(3):
    image_arr = np.array(image.split()[i])
    blurred_arr = np.array(blurred.split()[i])
    # 前景保持原样，背景使用模糊版本

result = Image.new('RGB', image.size)
result.save('blurred_background.jpg')
```

---

## 🎯 脚本详细说明

### 1️⃣ simple_inference.py（简单推理）

**适用场景**：单张图片或快速测试

```bash
python simple_inference.py <输入图片> [输出路径] [选项]

选项:
  --ckpt PATH      模型权重路径（默认: pretrained/modnet_photographic_portrait_matting.ckpt）
  --device DEVICE  cuda 或 cpu（默认: 自动检测）
  --ref-size SIZE  参考大小，默认512，越大越精细但更慢
```

**输出**：单个PNG灰度图

**示例**：
```bash
python simple_inference.py photo.jpg output_matte.png --ref-size 768
```

---

### 2️⃣ run_portrait_matting.py（批量推理）

**适用场景**：处理大量图片

```bash
python run_portrait_matting.py \
  --input-path <输入文件夹> \
  --output-path <输出文件夹> \
  [--ckpt PATH] \
  [--ref-size SIZE] \
  [--device DEVICE]

必需参数:
  --input-path PATH   输入图片文件夹
  --output-path PATH  输出结果文件夹

可选参数:
  --ckpt PATH         模型文件路径（默认值已设置）
  --ref-size INT      参考大小（默认512）
  --device DEVICE     计算设备（默认自动）
```

**特点**：
- 自动识别文件夹内所有图片
- 支持格式：JPG、PNG、BMP、GIF
- 进度显示
- 出错继续处理

**示例**：
```bash
python run_portrait_matting.py --input-path "D:\photos" --output-path "D:\results"
```

---

### 3️⃣ check_environment.py（环境检查）

验证所有依赖和配置是否正确。

```bash
python check_environment.py
```

**检查内容**：
- ✅ Python版本 >= 3.6
- ✅ PyTorch安装
- ✅ CUDA可用性
- ✅ 依赖包（pillow, numpy等）
- ✅ MODNet代码文件
- ✅ 模型文件存在
- ✅ 模型能否正常加载

---

## 🔧 参数优化指南

### ref-size 参数（处理大小）

| 值 | 处理速度 | 质量 | 内存占用 | 推荐场景 |
|----|--------|------|--------|--------|
| 256 | 快速 | 一般 | 少 | 快速演示、小屏幕 |
| 512 | 均衡 | 中等 | 中 | **默认值** |
| 768 | 较慢 | 较好 | 多 | 高质量需求 |
| 1024 | 慢 | 优秀 | 很多 | 专业应用 |

```bash
# 快速处理
python simple_inference.py photo.jpg --ref-size 256

# 高质量输出
python simple_inference.py photo.jpg --ref-size 1024
```

### device 参数（计算设备）

```bash
# 自动选择（默认）
python simple_inference.py photo.jpg --device auto

# 强制使用GPU
python simple_inference.py photo.jpg --device cuda

# 强制使用CPU
python simple_inference.py photo.jpg --device cpu
```

---

## 📈 性能参考

### GPU 性能 (NVIDIA RTX 3060)

| 输入分辨率 | 处理时间 | 质量 |
|-----------|--------|------|
| 512x512   | ~20ms  | 一般 |
| 1024x1024 | ~40ms  | 较好 |
| 2048x2048 | ~120ms | 优秀 |

### CPU 性能 (Intel i7)

| 输入分辨率 | 处理时间 | 质量 |
|-----------|--------|------|
| 512x512   | ~200ms | 一般 |
| 1024x1024 | ~500ms | 较好 |
| 2048x2048 | ~1500ms| 优秀 |

---

## 📂 项目结构

```
e:\debug\MODNet
├── pretrained/
│   ├── modnet_photographic_portrait_matting.ckpt  ← 预训练模型
│   └── README.md
├── src/
│   ├── models/
│   │   ├── modnet.py  ← MODNet模型定义
│   │   └── backbones/
│   │       └── mobilenetv2.py  ← 主干网络
│   └── trainer.py
├── demo/
│   ├── image_matting/
│   │   └── colab/inference.py  ← Colab演示
│   └── video_matting/
├── simple_inference.py         ← ✨ 简单推理脚本
├── run_portrait_matting.py     ← ✨ 批量推理脚本
├── check_environment.py        ← ✨ 环境检查脚本
├── USAGE_GUIDE.md             ← 这个文件
└── README.md                  ← 项目说明
```

---

## 🔍 输出解释

- **输出格式**: PNG灰度图像
- **像素值**: 0-255 (0=完全透明, 255=完全不透明)
- **命名规则**: `<输入名>_matte.png`

例如：`photo.jpg` → `photo_matte.png`

---

## 🐛 常见问题

### Q1: 没有GPU怎么办？
**A**: 脚本会自动检测，使用CPU推理（较慢）。运行时无需修改参数。

### Q2: 推理速度太慢？
**A**: 
- 减小 `--ref-size` (默认512，可改为256或384)
- 使用GPU加速 (`--device cuda`)
- 降低输入图像分辨率

### Q3: 输出质量不好？
**A**:
- 增加 `--ref-size` (改为768或1024)
- 确保输入图像清晰，最好是专业肖像照
- 模型针对专业肖像照优化，可能对其他照片效果一般

### Q4: 模型文件不存在？
**A**: 确保 `pretrained/modnet_photographic_portrait_matting.ckpt` 存在。
如果丢失，可从[Google Drive](https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing)下载。

### Q5: 运行报错 "ModuleNotFoundError: No module named 'torch'"
**A**: 安装PyTorch
```bash
pip install torch torchvision
# 或GPU版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q6: "CUDA out of memory"（显存不足）
**A**:
- 减小 `--ref-size` 参数
- 使用 `--device cpu` 改用CPU
- 关闭其他GPU程序

### Q7: 能否处理视频？
**A**: 可以，参考 `demo/video_matting/` 文件夹中的脚本。

---

## 🎨 高级用法

### Python代码集成

#### 方式1: 命令行集成

```python
from PIL import Image
import torch
from src.models.modnet import MODNet
import torchvision.transforms as transforms
import torch.nn.functional as F

# 初始化模型
modnet = MODNet(backbone_pretrained=False)
modnet = torch.nn.DataParallel(modnet)
weights = torch.load('pretrained/modnet_photographic_portrait_matting.ckpt')
modnet.load_state_dict(weights)
modnet.cuda()
modnet.eval()

# 准备图像
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
im = Image.open('portrait.jpg')
im = transform(im)[None, :, :, :]  # [1, 3, H, W]

# 推理
with torch.no_grad():
    _, _, matte = modnet(im.cuda(), True)
    # matte shape: [1, 1, H, W], 值范围 [0, 1]
```

#### 方式2: 自定义类封装

```python
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from src.models.modnet import MODNet

class PortraitMatter:
    def __init__(self, ckpt_path, device='cuda'):
        self.device = device
        self.model = MODNet(backbone_pretrained=False)
        self.model = torch.nn.DataParallel(self.model)
        
        weights = torch.load(ckpt_path, map_location=device)
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def process(self, image_path, ref_size=512):
        """处理单张图片"""
        image = Image.open(image_path)
        h, w = image.size
        
        # 预处理
        im = self.transform(image)[None, :, :, :]
        
        # 调整大小
        if im.shape[2] > ref_size or im.shape[3] > ref_size:
            im = F.interpolate(im, size=(ref_size, ref_size), mode='area')
        
        # 推理
        with torch.no_grad():
            im = im.to(self.device)
            _, _, matte = self.model(im, True)
        
        # 恢复原始大小
        matte = F.interpolate(matte, size=(h, w), mode='area')
        return matte[0][0].cpu().numpy()

# 使用示例
matter = PortraitMatter('pretrained/modnet_photographic_portrait_matting.ckpt')
matte = matter.process('photo.jpg')
```

---

## 📖 论文和引用

如果你使用了MODNet，请引用以下论文：

```bibtex
@article{MODNet2021,
  author    = {Ke, Zhanghan and Li, Kaican and Zhou, Yunmiao and Wu, Qiulin and Bao, Bingbing and Zhang, Wei and Sun, Mingming},
  title     = {MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition},
  journal   = {AAAI 2022},
  month     = {February},
  year      = {2022}
}
```

### 相关资源

- 📄 论文: [MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition](https://arxiv.org/pdf/2011.11961.pdf)
- 🎥 在线演示: https://zhke.io/#/?modnet_demo
- 📺 补充视频: https://youtu.be/PqJ3BRHX3Lc
- 💾 其他格式: ONNX/TorchScript版本可用

---

## 🎓 推荐使用流程

1. ✅ 运行 `python check_environment.py` 验证环境
2. ✅ 用 `python simple_inference.py <图片>` 测试单张图片
3. ✅ 用 `python run_portrait_matting.py ...` 批量处理
4. 🔄 将matte用于你的项目（背景替换、虚化等）
5. 📖 阅读论文了解算法原理

---

**祝使用愉快！有问题欢迎反馈。** 🚀
