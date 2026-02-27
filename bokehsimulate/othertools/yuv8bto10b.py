#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YUV420 8bit to 10bit Converter
将YUV420 8bit文件转换为YUV420 10bit文件
"""

import numpy as np
import argparse
import os


def yuv420_8bit_to_10bit(input_file, output_file, width, height, bit_shift=2):
    """
    将YUV420 8bit文件转换为YUV420 10bit文件
    
    参数:
        input_file: 输入的8bit YUV文件路径
        output_file: 输出的10bit YUV文件路径
        width: 视频宽度
        height: 视频高度
        bit_shift: 位移量,默认为2 (8bit -> 10bit需要左移2位)
    """
    
    # 计算每帧的大小
    y_size = width * height  # Y分量大小
    uv_size = (width // 2) * (height // 2)  # U和V分量大小(各为Y的1/4)
    frame_size_8bit = y_size + 2 * uv_size  # 一帧8bit数据的总大小
    
    # 检查输入文件大小
    file_size = os.path.getsize(input_file)
    num_frames = file_size // frame_size_8bit
    
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"分辨率: {width}x{height}")
    print(f"文件大小: {file_size} 字节")
    print(f"总帧数: {num_frames}")
    print(f"开始转换...\n")
    
    with open(input_file, 'rb') as fin, open(output_file, 'wb') as fout:
        frame_count = 0
        
        while True:
            # 读取一帧的8bit数据
            frame_data = fin.read(frame_size_8bit)
            
            if len(frame_data) < frame_size_8bit:
                break
            
            # 将8bit数据转换为numpy数组
            frame_8bit = np.frombuffer(frame_data, dtype=np.uint8)
            
            # 转换为16bit并左移(8bit->10bit需要左移2位)
            # 例如: 8bit的255 -> 10bit的1020 (255 << 2 = 1020)
            frame_10bit = frame_8bit.astype(np.uint16) << bit_shift
            
            # 写入10bit数据(使用16bit存储)
            fout.write(frame_10bit.tobytes())
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"已处理 {frame_count}/{num_frames} 帧", end='\r')
    
    print(f"\n转换完成! 共处理 {frame_count} 帧")
    print(f"输出文件大小: {os.path.getsize(output_file)} 字节")


def main():
    parser = argparse.ArgumentParser(
        description='将YUV420 8bit文件转换为YUV420 10bit文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python yuv_converter.py input.yuv output.yuv 1920 1080
  python yuv_converter.py input.yuv output.yuv 3840 2160 --bit-shift 2
        """)
    
    parser.add_argument('input', help='输入的8bit YUV文件')
    parser.add_argument('output', help='输出的10bit YUV文件')
    parser.add_argument('width', type=int, help='视频宽度')
    parser.add_argument('height', type=int, help='视频高度')
    parser.add_argument('--bit-shift', type=int, default=2,
                        help='位移量 (默认: 2, 即8bit->10bit)')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在!")
        return
    
    # 检查分辨率是否有效
    if args.width <= 0 or args.height <= 0:
        print(f"错误: 无效的分辨率 {args.width}x{args.height}")
        return
    
    # 执行转换
    yuv420_8bit_to_10bit(args.input, args.output, args.width, args.height, args.bit_shift)


if __name__ == '__main__':
    main()