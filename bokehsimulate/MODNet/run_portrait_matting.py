"""
MODNet 人像Matting推理脚本
用法: python run_portrait_matting.py --input-path <输入图片文件夹> --output-path <输出文件夹>
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


def main():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description='MODNet 人像Matting推理')
    parser.add_argument('--input-path', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output-path', type=str, required=True, help='输出结果文件夹路径')
    parser.add_argument('--ckpt-path', type=str, 
                       default='pretrained/modnet_photographic_portrait_matting.ckpt',
                       help='预训练模型路径')
    parser.add_argument('--ref-size', type=int, default=512, help='参考大小')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='使用设备: cuda 或 cpu')
    args = parser.parse_args()

    # 检查输入参数
    if not os.path.exists(args.input_path):
        print(f'错误: 找不到输入文件夹: {args.input_path}')
        sys.exit(1)
    
    if not os.path.exists(args.output_path):
        print(f'创建输出文件夹: {args.output_path}')
        os.makedirs(args.output_path, exist_ok=True)
    
    if not os.path.exists(args.ckpt_path):
        print(f'错误: 找不到模型文件: {args.ckpt_path}')
        print(f'请确保模型文件位于 {args.ckpt_path}')
        sys.exit(1)

    print(f'使用设备: {args.device}')
    print(f'模型文件: {args.ckpt_path}')
    
    # 定义图像转张量的变换
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建MODNet模型并加载权重
    print('加载模型...')
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if args.device == 'cuda':
        modnet = modnet.cuda()
        weights = torch.load(args.ckpt_path)
    else:
        weights = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    
    modnet.load_state_dict(weights)
    modnet.eval()
    print('模型加载完成！')

    # 推理图像
    im_names = [f for f in os.listdir(args.input_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
    
    if not im_names:
        print(f'错误: 在 {args.input_path} 中找不到图像文件')
        sys.exit(1)
    
    print(f'找到 {len(im_names)} 张图片，开始推理...\n')
    
    with torch.no_grad():
        for im_name in im_names:
            print(f'处理图片: {im_name}')
            
            # 读取图像
            im_path = os.path.join(args.input_path, im_name)
            im = Image.open(im_path)
            
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

            # 添加批处理维度
            im = im[None, :, :, :]

            # 根据参考大小调整输入图像
            im_b, im_c, im_h, im_w = im.shape
            if max(im_h, im_w) < args.ref_size or min(im_h, im_w) > args.ref_size:
                if im_w >= im_h:
                    im_rh = args.ref_size
                    im_rw = int(im_w / im_h * args.ref_size)
                else:
                    im_rw = args.ref_size
                    im_rh = int(im_h / im_w * args.ref_size)
            else:
                im_rh = im_h
                im_rw = im_w
            
            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im_resized = F.interpolate(im, size=(im_rh, im_rw), mode='area')

            # 推理
            if args.device == 'cuda':
                im_resized = im_resized.cuda()
            
            _, _, matte = modnet(im_resized, True)

            # 调整matte回原始大小并保存
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            
            matte_name = os.path.splitext(im_name)[0] + '_matte.png'
            matte_path = os.path.join(args.output_path, matte_name)
            Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(matte_path)
            print(f'✓ 已保存: {matte_path}\n')

    print('推理完成！')


if __name__ == '__main__':
    main()
