#!/usr/bin/env python3
"""
将非标准10bit YUV Semi-Planar格式转换为标准10bit YUV420格式
输入格式: w[1920]_h[1080]_stride[1920]_scanline[1088]
10bit格式: 每个像素占用2字节(16bit存储,高6bit为0)
"""

import sys
import struct

def convert_yuv_semiplanar_to_yuv420(input_file, output_file):
    """
    转换10bit YUV Semi-Planar到标准10bit YUV420格式
    
    参数说明:
    - width: 1920 (图像宽度,单位:像素)
    - height: 1080 (图像高度,单位:像素)
    - stride: 1920 (每行的像素数,包含padding)
    - scanline: 1088 (包含padding的总行数)
    - 10bit: 每个像素用2字节存储
    """
    
    # 参数设置
    width = 1920
    height = 1080
    stride = 1920
    scanline = 1088
    bytes_per_pixel = 2  # 10bit格式每个像素2字节
    
    # 计算各个plane的大小
    y_plane_size = stride * scanline  # Y平面大小
    uv_plane_size = stride * (scanline // 2)  # UV交织平面大小
    
    total_input_size = y_plane_size + uv_plane_size
    
    print(f"输入文件参数:")
    print(f"  宽度: {width}")
    print(f"  高度: {height}")
    print(f"  Stride: {stride}")
    print(f"  Scanline: {scanline}")
    print(f"  Y平面大小: {y_plane_size} bytes")
    print(f"  UV平面大小: {uv_plane_size} bytes")
    print(f"  总输入大小: {total_input_size} bytes")
    
    try:
        # 读取输入文件
        with open(input_file, 'rb') as f:
            data = f.read()
        
        print(f"\n实际文件大小: {len(data)} bytes")
        
        if len(data) < total_input_size:
            print(f"警告: 文件大小不足,期望至少 {total_input_size} bytes")
        
        # 提取Y平面 (去除padding)
        y_data = bytearray()
        for row in range(height):
            start = row * stride
            end = start + width
            y_data.extend(data[start:end])
        
        # 提取UV平面 (Semi-Planar格式是UV交织的)
        uv_start = y_plane_size
        u_data = bytearray()
        v_data = bytearray()
        
        for row in range(height // 2):
            row_start = uv_start + row * stride
            for col in range(width // 2):
                pixel_start = row_start + col * 2
                u_data.append(data[pixel_start])      # U分量
                v_data.append(data[pixel_start + 1])  # V分量
        
        # 写入标准YUV420格式 (Y-U-V planar)
        with open(output_file, 'wb') as f:
            f.write(y_data)
            f.write(u_data)
            f.write(v_data)
        
        output_size = len(y_data) + len(u_data) + len(v_data)
        expected_size = width * height * 3 // 2
        
        print(f"\n转换完成!")
        print(f"  Y平面: {len(y_data)} bytes")
        print(f"  U平面: {len(u_data)} bytes")
        print(f"  V平面: {len(v_data)} bytes")
        print(f"  输出总大小: {output_size} bytes")
        print(f"  期望大小: {expected_size} bytes")
        print(f"\n输出文件: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("用法: python yuv_converter.py <input_file> [output_file]")
        print("\n示例:")
        print("  python yuv_converter.py input.yuv output.yuv")
        print("  python yuv_converter.py input.yuv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 如果没有指定输出文件名,自动生成
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        if input_file.endswith('.yuv'):
            output_file = input_file.replace('.yuv', '_420.yuv')
        else:
            output_file = input_file + '_420.yuv'
    
    convert_yuv_semiplanar_to_yuv420(input_file, output_file)

if __name__ == "__main__":
    main()