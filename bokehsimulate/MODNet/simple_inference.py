"""
简单的单图推理演示 - 快速测试脚本
用法: python simple_inference.py <输入图片路径> [输出路径]
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.models.modnet import MODNet


def infer_single_image(image_path, output_path, ckpt_path, device='cuda', ref_size=512):
    """
    对单张图片进行推理
    
    Args:
        image_path: 输入图片路径
        output_path: 输出matte路径
        ckpt_path: 模型权重路径
        device: 计算设备 ('cuda' 或 'cpu')
        ref_size: 参考大小
    
    Returns:
        matte: numpy数组，值范围[0, 1]
    """
    # 定义转换
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 读取图像
    print(f'📖 读取图片: {image_path}')
    im = Image.open(image_path)
    im_size = im.size
    print(f'   图片尺寸: {im_size}')
    
    # 统一图像通道为3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # 转换为PyTorch张量
    im = Image.fromarray(im)
    im = im_transform(im)
    im = im[None, :, :, :]  # [1, 3, H, W]
    
    im_b, im_c, im_h, im_w = im.shape

    # 调整大小以匹配参考大小
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        else:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    print(f'   调整大小: ({im_h}, {im_w}) -> ({im_rh}, {im_rw})')
    
    im_resized = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # 创建模型并加载权重
    print(f'🤖 加载模型: {ckpt_path}')
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if device == 'cuda' and torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
        print(f'   设备: GPU (CUDA)')
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        device = 'cpu'
        print(f'   设备: CPU')
    
    modnet.load_state_dict(weights)
    modnet.eval()

    # 推理
    print(f'🔄 推理中...')
    with torch.no_grad():
        if device == 'cuda':
            im_resized = im_resized.cuda()
        _, _, matte = modnet(im_resized, True)

    # 恢复原始大小
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(output_path)
    
    print(f'✅ 成功保存: {output_path}')
    print(f'   Matte范围: [{matte.min():.3f}, {matte.max():.3f}]')
    
    return matte


def main():
    parser = argparse.ArgumentParser(description='MODNet - 简单推理演示')
    parser.add_argument('input', type=str, nargs='?', default='pics/0.jpg', help='输入图片路径')
    parser.add_argument('output', type=str, nargs='?', default=None, help='输出matte路径（可选）')
    parser.add_argument('--ckpt', type=str, default='pretrained/modnet_photographic_portrait_matting.ckpt',help='模型权重路径')
    parser.add_argument('--device', type=str, default='auto',help='计算设备: cuda/cpu/auto')
    parser.add_argument('--ref-size', type=int, default=512, help='参考大小')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f'❌ 错误: 找不到输入文件 {args.input}')
        sys.exit(1)
    
    # 确定输出路径
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join('resultMatting', f'{base_name}_matte.png')
    elif not args.output.endswith(('.png', '.jpg', '.jpeg')):
        args.output = f'{args.output}_matte.png'
    
    # 检查模型文件
    if not os.path.exists(args.ckpt):
        print(f'❌ 错误: 找不到模型文件 {args.ckpt}')
        sys.exit(1)
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"使用设备: {device}")
    
    print(f'\n{"="*50}')
    print(f'MODNet 人像Matting推理')
    print(f'{"="*50}\n')
    
    try:
        matte = infer_single_image(args.input, args.output, args.ckpt, device, args.ref_size)
        print(f'\n{"="*50}')
        print(f'推理完成！')
        print(f'{"="*50}\n')
    except Exception as e:
        print(f'\n❌ 推理失败: {str(e)}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
