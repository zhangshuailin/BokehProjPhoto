#!/usr/bin/env python3
"""
RGB图片(JPG/PNG)转YUV420格式转换工具

输出文件自动以尺寸命名，如：photo_1920x1080.yuv

使用方法：
    python rgb_to_yuv420.py input.jpg
    python rgb_to_yuv420.py input.png
"""

import sys
import os
import numpy as np
from PIL import Image


def rgb_to_yuv(rgb_image):
    """将RGB图像转换为YUV颜色空间（BT.601标准）"""
    rgb = np.array(rgb_image, dtype=np.float32)
    
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128
    v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128
    
    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255).astype(np.uint8)
    v = np.clip(v, 0, 255).astype(np.uint8)
    
    return y, u, v


def downsample_chroma(u, v):
    """对色度分量进行4:2:0下采样"""
    height, width = u.shape
    new_height = height // 2
    new_width = width // 2
    
    u_downsampled = np.zeros((new_height, new_width), dtype=np.uint8)
    v_downsampled = np.zeros((new_height, new_width), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            u_block = u[i*2:i*2+2, j*2:j*2+2]
            v_block = v[i*2:i*2+2, j*2:j*2+2]
            u_downsampled[i, j] = np.mean(u_block).astype(np.uint8)
            v_downsampled[i, j] = np.mean(v_block).astype(np.uint8)
    
    return u_downsampled, v_downsampled


def convert_to_yuv420(input_path):
    """将JPG/PNG转换为YUV420格式，输出文件名包含尺寸"""
    
    # 读取图片
    img = Image.open(input_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # 确保尺寸为偶数
    if width % 2 != 0 or height % 2 != 0:
        width = width - (width % 2)
        height = height - (height % 2)
        img = img.resize((width, height), Image.LANCZOS)
    
    # 生成输出文件名：原文件名_宽x高.yuv
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.dirname(input_path) or '.'
    output_path = os.path.join(output_dir, f"{base_name}_{width}x{height}.yuv")
    
    # 转换
    y, u, v = rgb_to_yuv(img)
    u_420, v_420 = downsample_chroma(u, v)
    
    # 写入文件
    with open(output_path, 'wb') as f:
        f.write(y.tobytes())
        f.write(u_420.tobytes())
        f.write(v_420.tobytes())
    
    print(f"输入: {input_path}")
    print(f"尺寸: {width}x{height}")
    print(f"输出: {output_path}")
    print(f"\n播放: ffplay -f rawvideo -pix_fmt yuv420p -s {width}x{height} {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python rgb_to_yuv420.py <图片.jpg/png>")
        sys.exit(1)
    
    convert_to_yuv420(sys.argv[1])


if __name__ == "__main__":
    main()