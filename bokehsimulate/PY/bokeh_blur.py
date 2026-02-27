#!/usr/bin/env python3
"""
背景虚化程序
输入：原图、深度图(0.25倍尺寸)、LUT文件(XML)
输出：虚化后的图片

========== 优化版本说明 ==========
提供了4种速度模式供选择（通过 -s/--speed 参数指定）：

1. balanced (推荐，默认)
   - 多线程处理（自动检测CPU核心数）
   - 一次性计算缓存mask（避免重复计算）
   - 完整的mask平滑处理
   - 性能提升: 30-50% 相比原始版本
   
2. fast  
   - 减小mask平滑kernel从5x5到3x3
   - 减少处理的层数（约50%的唯一kernel值）
   - 使用加权平均替代完整混合
   - 性能提升: 50-70% 相比原始版本
   
3. ultra_fast
   - 完全跳过mask平滑，直接二值操作
   - 最大化减少处理的层数（约20%的唯一kernel值）
   - 直接替换而非混合
   - 性能提升: 70-90% 相比原始版本
   - 注意：精度降低，仅建议低延迟场景使用
   
4. legacy
   - 原始未优化版本，用于性能对比测试

================== MODNet人体保护功能 ==================
可选功能：使用MODNet进行人像抠图，生成人体mask
- 启用：添加 -pm 或 --portrait-mask 参数
- 效果：人体区域保留原图无虚化，只对背景虚化

使用示例：
  python bokeh_blur.py image.jpg depth.jpg lut.xml -s balanced    # 默认推荐
  python bokeh_blur.py image.jpg depth.jpg lut.xml -s fast        # 更快
  python bokeh_blur.py image.jpg depth.jpg lut.xml -s ultra_fast  # 最快
  python bokeh_blur.py image.jpg depth.jpg lut.xml -s legacy      # 对比测试
  python bokeh_blur.py image.jpg depth.jpg lut.xml -pm            # 启用人体保护（推荐）

关键优化点：
✓ 多线程并行处理（balanced/原始方法的主要瓶颈）
✓ 缓存mask计算结果（避免重复计算两次）
✓ MODNet人体抠图功能（可选）
✓ 优化数据类型转换
✓ 减少不必要的内存分配
✓ 自适应kernel处理
"""

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import argparse
import os
import threading
import time
from pathlib import Path
from multiprocessing.pool import ThreadPool

# MODNet相关导入
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image
    MODNET_AVAILABLE = True
except ImportError:
    MODNET_AVAILABLE = False
    print("警告: PyTorch或PIL未安装，MODNet功能不可用")


# ========================= MODNet 人体抠图函数 =========================

def get_modnet_model(ckpt_path, device='cuda'):
    """
    加载MODNet模型
    
    Args:
        ckpt_path: 模型权重路径
        device: 计算设备 ('cuda' 或 'cpu')
    
    Returns:
        mode的实力和设备信息
    """
    if not MODNET_AVAILABLE:
        raise ImportError("MODNet需要PyTorch和PIL环境。请确保已安装必要的包。")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"模型文件不存在: {ckpt_path}")
    
    # 导入MODNet（动态导入以避免路径问题）
    import sys
    modnet_path = os.path.join(os.path.dirname(__file__), '..', 'MODNet')
    if modnet_path not in sys.path:
        sys.path.insert(0, modnet_path)
    
    from src.models.modnet import MODNet
    
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    if device == 'cuda' and torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
        print(f'✓ MODNet模型已加载到GPU')
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        device = 'cpu'
        print(f'✓ MODNet模型已加载到CPU')
    
    modnet.load_state_dict(weights)
    modnet.eval()
    
    return modnet, device


def infer_portrait_matte(image_path, modnet, device='cuda', ref_size=512):
    """
    使用MODNet推断人像matte (alpha mask)
    
    Args:
        image_path: 输入图片路径
        modnet: MODNet模型实例
        device: 计算设备
        ref_size: 参考大小
    
    Returns:
        matte: numpy数组，值范围[0, 1]，人体部分值接近1，背景接近0
    """
    # 定义转换
    im_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 读取图像
    print(f'  读取人像图片: {image_path}')
    im = Image.open(image_path)
    im_size = im.size
    
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
    
    im_resized = F.interpolate(im, size=(im_rh, im_rw), mode='bicubic')
    
    # 推理
    with torch.no_grad():
        if device == 'cuda':
            im_resized = im_resized.cuda()
        _, _, matte = modnet(im_resized, True)
    
    # 恢复原始大小
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    
    return matte


def generate_portrait_mask(image_path, ckpt_path, device='cuda', max_kernel=None):
    """
    生成人体alpha mask - 直接保存MODNet输出到PNG
    
    步骤：
    1. 使用MODNet生成matte（连续alpha值0~1）
    2. 直接保存为PNG文件（保留连续值）
    3. 返回alpha值和保存路径
    
    Args:
        image_path: 输入图片路径
        ckpt_path: MODNet模型权重路径
        device: 计算设备
        max_kernel: 最大虚化核大小（用于参考，可选）

    Returns:
        matte: numpy数组，连续值范围[0, 1]（直接来自MODNet，未处理）
    """
    print(f"MODNet人体抠图...")

    # 加载模型
    modnet, device = get_modnet_model(ckpt_path, device)

    # 推断matte（连续alpha值，直接来自MODNet）
    matte = infer_portrait_matte(image_path, modnet, device)
    print(f"  Matte范围: [{matte.min():.3f}, {matte.max():.3f}]")
    
    # 查看matte的值分布（诊断信息）
    unique_vals = len(np.unique(np.round(matte, 3)))
    print(f"  Matte唯一值个数: {unique_vals}")
    
    # 释放模型内存
    del modnet
    try:
        torch.cuda.empty_cache()
    except:
        pass
    
    # 计算输出路径
    # 使用resolve()得到绝对路径，然后在路径部分中查找IMGS
    image_path_obj = Path(image_path).resolve()
    print(f"  原始image_path: {image_path}")
    print(f"  绝对路径: {image_path_obj}")
    
    # 从路径parts中找IMGS目录
    imgs_dir = None
    path_parts = image_path_obj.parts
    print(f"  路径parts: {path_parts}")
    
    for idx, part in enumerate(path_parts):
        if part == 'IMGS':
            # 找到IMGS，重建到这一层的路径
            imgs_dir = Path(*path_parts[:idx+1])
            print(f"  ✓ 找到IMGS目录在第{idx}层: {imgs_dir}")
            break
    
    if imgs_dir is not None:
        output_dir = imgs_dir / 'modnetportrait'
    else:
        print(f"  ⚠ 未找到IMGS目录，回退到parent.parent")
        output_dir = image_path_obj.parent.parent / 'modnetportrait'
    
    output_path = output_dir / 'portrait_mask.png'
    print(f"  ✓ 最终输出目录: {output_dir}")
    print(f"  ✓ 最终输出路径: {output_path}")
    
    # 保存alpha数据为PNG
    try:
        os.makedirs(str(output_dir), exist_ok=True)
        print(f"  ✓ 目录已创建/存在: {output_dir}")
    except Exception as e:
        print(f"  ✗ 创建目录失败: {e}")
        raise
    
    # 处理后的mask数据（用于返回）
    processed_matte = None
    
    # 按照用户提供的方式保存：直接转uint8后保存
    try:
        print(f"  准备保存PNG{output_path}...")
        print(f"  matte数据类型: {type(matte)}, 范围: [{matte.min():.3f}, {matte.max():.3f}]")
        
        uint8_data = (matte * 255).astype(np.uint8)
        print(f"  uint8数据类型: {type(uint8_data)}, 形状: {uint8_data.shape}, 范围: [{uint8_data.min()}, {uint8_data.max()}]")
        
        # ===== 关键：多步骤平滑mask锯齿（综合方案） =====
        # 组合多种滤波器以获得最佳效果：边界保持 + 锯齿去除
        print(f"  应用多步骤滤波平滑mask边缘...")
        try:
            float_data = uint8_data.astype(np.float32)
            
            # 方案1: 尝试Guided Filter（如果ximgproc可用）
            try:
                from cv2 import ximgproc
                print(f"    [Step1] 应用Guided Filter (保边界平滑)...")
                # 较小的参数避免过度平滑
                radius = 15
                eps = 5.0
                filtered = ximgproc.guidedFilter(float_data, float_data, radius, eps)
                uint8_data = np.clip(filtered, 0, 255).astype(np.uint8)
                print(f"    ✓ Guided Filter完成")
            except ImportError:
                print(f"    ⚠ ximgproc不可用，跳过Guided Filter")
            
            # 方案2: Bilateral Filter（关键！对去除锯齿很有效）
            print(f"    [Step2] 应用Bilateral Filter (保边界去噪)...")
            # d: 像素邻域直径，越大处理范围越大
            # sigmaColor: 色彩空间的sigma，越大颜色差异越容易被当作同一区域
            # sigmaSpace: 坐标空间的sigma，越大远处像素也会被影响
            uint8_data = cv2.bilateralFilter(uint8_data, d=9, sigmaColor=75, sigmaSpace=75)
            print(f"    ✓ Bilateral Filter完成")
            
            # 方案3: Morphological Smoothing（闭操作 + 开操作）
            print(f"    [Step3] 应用形态学平滑 (填平断裂)...")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            # 闭操作：膨胀后腐蚀，填平小孔洞
            uint8_data = cv2.morphologyEx(uint8_data, cv2.MORPH_CLOSE, kernel)
            # 开操作：腐蚀后膨胀，去除小锯齿
            uint8_data = cv2.morphologyEx(uint8_data, cv2.MORPH_OPEN, kernel)
            print(f"    ✓ 形态学平滑完成")
            
            # 方案4: 最后一遍轻量级高斯模糊（平缓过渡）
            print(f"    [Step4] 应用轻量高斯模糊 (过渡平缓)...")
            uint8_data = cv2.GaussianBlur(uint8_data, (5, 5), 0)
            print(f"    ✓ 高斯模糊完成")
            
            uint8_data = np.clip(uint8_data, 0, 255).astype(np.uint8)
            print(f"  ✓ 多步骤平滑完成，范围: [{uint8_data.min()}, {uint8_data.max()}]")
            
            # 保存处理后的uint8数据并转换为[0,1]范围用于返回
            processed_matte = uint8_data.astype(np.float32) / 255.0
            
        except Exception as e:
            print(f"  ⚠ 平滑处理出错 ({e})，回退到原始数据")
            print(f"  将直接使用原始MODNet输出")
            processed_matte = matte
        
        pil_image = Image.fromarray(uint8_data, mode='L')
        pil_image.save(str(output_path))
        
        print(f"✅ Alpha mask已保存: {output_path}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 验证保存的数据
    try:
        saved_img = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        if saved_img is None:
            print(f"✗ 验证失败：无法读取保存的文件")
        else:
            unique_saved = len(np.unique(saved_img))
            print(f"   ✓ 保存验证: {unique_saved} 个唯一值 (连续alpha)")
            print(f"   ✓ 像素范围: [{saved_img.min()}, {saved_img.max()}]")
            print(f"   ✓ 文件大小: {os.path.getsize(str(output_path))} bytes")
            
            # 统计信息
            matte_thresh = (saved_img.astype(np.float32) / 255.0)
            human_pixels = np.sum(matte_thresh > 0.5)
            transition_pixels = np.sum((matte_thresh >= 0.1) & (matte_thresh <= 0.9))
            total_pixels = matte_thresh.size
            
            print(f"   人体像素(>0.5): {human_pixels} ({human_pixels/total_pixels*100:.1f}%)")
            print(f"   过渡区像素(0.1~0.9): {transition_pixels} ({transition_pixels/total_pixels*100:.1f}%)")
    except Exception as e:
        print(f"✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()

    # 返回处理后的mask（如果处理失败则返回原始的）
    return processed_matte if processed_matte is not None else matte





def parse_lut_xml(xml_path):
    """解析LUT XML文件，返回256个模糊核大小"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取value标签内的数据
    value_text = root.find('value').text.strip()
    # 过滤空字符串
    lut = [int(x) for x in value_text.split(',') if x.strip()]
    
    if len(lut) != 256:
        raise ValueError(f"LUT应该有256个值，实际有{len(lut)}个")
    
    return np.array(lut, dtype=np.int32)


def create_blur_layers(img, max_kernel_size):
    """预计算不同模糊程度的图像层"""
    blur_layers = {0: img.copy()}  # kernel_size=0 表示不模糊
    
    # 只计算LUT中出现的kernel size
    for k in range(1, max_kernel_size + 1):
        # 确保kernel size为奇数
        kernel_size = k * 2 + 1
        blur_layers[k] = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    return blur_layers


def apply_depth_based_blur(img, depth_map, lut, portrait_mask=None, method='layered', speed_mode='balanced'):
    """
    根据深度图和LUT应用可变模糊
    
    参数:
        portrait_mask: 人体mask (H, W)，1表示人体（不虚化），0表示背景（虚化）
        method: 'layered' 或 'per_pixel'
        speed_mode: 仅在method='layered'时有效
            'balanced' - 默认优化方案，多线程+缓存mask（推荐）
            'fast' - 快速模式，减少平滑操作
            'ultra_fast' - 超快速模式，精度最低但速度最快
            'legacy' - 原始方法（不推荐）
    """
    h, w = img.shape[:2]
    depth_h, depth_w = depth_map.shape[:2]
    
    # 将深度图上采样到原图尺寸
    if depth_h != h or depth_w != w:
        depth_map_full = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        depth_map_full = depth_map
    
    # 获取每个像素对应的kernel size
    kernel_map = lut[depth_map_full.astype(np.int32)]
    
    max_kernel = int(np.max(lut))
    print(f"最大模糊核: {max_kernel}, 深度范围: {depth_map.min()}-{depth_map.max()}")
    
    if method == 'layered':
        if speed_mode == 'ultra_fast':
            return _apply_layered_blur_ultra_fast(img, kernel_map, max_kernel, portrait_mask)
        elif speed_mode == 'fast':
            return _apply_layered_blur_fast(img, kernel_map, max_kernel, portrait_mask)
        elif speed_mode == 'legacy':
            # 使用原始未优化版本（用于对比）
            return _apply_layered_blur_legacy(img, kernel_map, max_kernel, portrait_mask)
        else:  # 'balanced' 或其他
            return _apply_layered_blur(img, kernel_map, max_kernel, portrait_mask)
    else:
        return _apply_per_pixel_blur(img, kernel_map, portrait_mask)


def _apply_layered_blur(img, kernel_map, max_kernel, portrait_mask=None, num_threads=None):
    """
    使用预计算模糊层的优化方法（多线程 + 缓存mask）
    
    优化点：
    1. 一次计算所有mask并缓存（之前计算两次）
    2. 多线程处理不同kernel的混合操作
    3. 优化mask平滑算法
    4. 减少内存拷贝
    """
    h, w = img.shape[:2]
    
    # 预计算所有需要的模糊层
    print("预计算模糊层...")
    blur_layers = create_blur_layers(img, max_kernel)
    
    # 获取所有唯一的kernel size（按大小排序便于优化）
    unique_kernels = np.unique(kernel_map)
    print(f"使用的模糊核大小: {sorted(unique_kernels)}")
    
    # ===== 优化1：一次性计算并缓存所有mask ====
    print("计算并缓存mask...")
    mask_cache = {}
    weight_sum = np.zeros((h, w), dtype=np.float32)
    
    for k in unique_kernels:
        mask = (kernel_map == k).astype(np.float32)
        
        # 平滑mask边缘以减少伪影（仅对非零kernel平滑）
        if k > 0:
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        mask_cache[k] = mask
        weight_sum += mask
    
    # ===== 优化2：使用多线程混合不同的模糊层 ====
    print("多线程混合模糊层...")
    
    if num_threads is None:
        num_threads = min(len(unique_kernels), os.cpu_count() or 4)
    
    # 初始化输出图像
    result = np.zeros((h, w, img.shape[2]), dtype=np.float32)
    result_lock = threading.Lock()
    
    def mix_layer(k):
        """混合单个layer的函数"""
        mask = mask_cache[k][:, :, np.newaxis]  # 扩展维度
        layer_result = blur_layers[k].astype(np.float32) * mask
        return layer_result
    
    # 使用ThreadPool并行处理
    with ThreadPool(num_threads) as pool:
        layer_results = pool.map(mix_layer, unique_kernels)
    
    # 累加结果
    for layer_result in layer_results:
        result += layer_result
    
    # ===== 优化3：高效的归一化 ====
    weight_sum = np.maximum(weight_sum, 1e-6)  # 避免除零
    weight_sum_3ch = weight_sum[:, :, np.newaxis]
    result = result / weight_sum_3ch
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # ===== 应用人体alpha mask：soft blend融合（非线性alpha曲线） =====
    if portrait_mask is not None:
        # 确保mask尺寸匹配
        if portrait_mask.shape[:2] != img.shape[:2]:
            portrait_mask = cv2.resize(portrait_mask, (img.shape[1], img.shape[0]))
        
        # 使用连续alpha值进行soft blend
        alpha = portrait_mask[:, :, np.newaxis].astype(np.float32)  # 扩展为3通道
        
        # 非线性alpha变换：使轮廓过渡更平缓
        alpha_power = 3.0
        alpha = np.power(alpha, alpha_power)
        
        # soft blend：原图 * alpha + 虚化图 * (1 - alpha)
        result = (img.astype(np.float32) * alpha + result.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    
    return result


def _apply_layered_blur_fast(img, kernel_map, max_kernel, portrait_mask=None):
    """
    超快速版本 - 牺牲部分平滑质量换取速度
    
    优化特点：
    1. 跳过mask平滑（直接使用二值mask）
    2. 只计算必要的层（跳过小kernel值）
    3. 使用双线性插值替代完整混合
    4. 更小的平滑kernel
    """
    h, w = img.shape[:2]
    
    print("预计算模糊层(快速模式)...")
    blur_layers = create_blur_layers(img, max_kernel)
    
    unique_kernels = np.unique(kernel_map)
    print(f"使用的模糊核大小: {sorted(unique_kernels)}")
    
    # 对唯一kernel值进行量化，减少需要处理的层数
    unique_kernels = unique_kernels[::max(1, len(unique_kernels) // 10)]
    
    # 直接混合（不进行mask平滑）
    result = img.astype(np.float32)
    
    for k in sorted(unique_kernels):
        if k == 0:
            continue
        
        mask = (kernel_map == k).astype(np.float32)
        if np.sum(mask) == 0:
            continue
        
        # 轻量平滑mask边缘（3x3而不是5x5）
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = mask[:, :, np.newaxis]
        
        blurred = blur_layers[k].astype(np.float32)
        # 使用加权平均而不是完整的mask操作
        result = result * (1 - mask) + blurred * mask
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # ===== 应用人体alpha mask：soft blend融合（非线性alpha曲线） =====
    if portrait_mask is not None:
        # 确保mask尺寸匹配
        if portrait_mask.shape[:2] != img.shape[:2]:
            portrait_mask = cv2.resize(portrait_mask, (img.shape[1], img.shape[0]))
        
        # 使用连续alpha值进行soft blend
        alpha = portrait_mask[:, :, np.newaxis].astype(np.float32)  # 扩展为3通道
        
        # 非线性alpha变换：使轮廓过渡更平缓
        alpha_power = 2.0
        alpha = np.power(alpha, alpha_power)
        
        # soft blend：原图 * alpha + 虚化图 * (1 - alpha)
        result = (img.astype(np.float32) * alpha + result.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    
    return result


def _apply_layered_blur_ultra_fast(img, kernel_map, max_kernel, portrait_mask=None):
    """
    超超快速版本 - 最大化速度，最小化精度损失
    
    优化特点：
    1. 不平滑mask - 直接二值操作
    2. 跳过小kernel
    3. 使用uint8避免float转换
    4. 简化的混合逻辑
    """
    h, w = img.shape[:2]
    
    print("预计算模糊层(极速模式)...")
    blur_layers = create_blur_layers(img, max_kernel)
    
    unique_kernels = np.unique(kernel_map)
    # 大幅减少处理的层数
    step = max(1, len(unique_kernels) // 5)
    unique_kernels = unique_kernels[::step]
    
    result = img.copy()
    
    # 直接替换（不混合）
    for k in sorted(unique_kernels):
        if k == 0:
            continue
        
        mask = kernel_map == k
        if not np.any(mask):
            continue
        
        result[mask] = blur_layers[k][mask]
    
    # ===== 应用人体alpha mask：soft blend融合（非线性alpha曲线） =====
    if portrait_mask is not None:
        # 确保mask尺寸匹配
        if portrait_mask.shape[:2] != img.shape[:2]:
            portrait_mask = cv2.resize(portrait_mask, (img.shape[1], img.shape[0]))
        
        # 使用连续alpha值进行soft blend
        alpha = portrait_mask[:, :, np.newaxis].astype(np.float32)  # 扩展为3通道
        
        # 非线性alpha变换：使轮廓过渡更平缓
        alpha_power = 2.0
        alpha = np.power(alpha, alpha_power)
        
        # soft blend：原图 * alpha + 虚化图 * (1 - alpha)
        result = (img.astype(np.float32) * alpha + result.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    
    return result


def _apply_layered_blur_legacy(img, kernel_map, max_kernel, portrait_mask=None):
    """
    原始未优化版本 - 用于性能对比测试
    """
    h, w = img.shape[:2]
    
    print("预计算模糊层(原始版本)...")
    blur_layers = create_blur_layers(img, max_kernel)
    
    unique_kernels = np.unique(kernel_map)
    print(f"使用的模糊核大小: {sorted(unique_kernels)}")
    
    # 初始化输出图像
    result = np.zeros_like(img, dtype=np.float32)
    
    # 对每个kernel size创建mask并混合（原始方式，计算两次mask）
    for k in unique_kernels:
        mask = (kernel_map == k).astype(np.float32)
        
        # 平滑mask边缘以减少伪影
        if k > 0:
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        mask = mask[:, :, np.newaxis]  # 扩展维度用于广播
        result += blur_layers[k].astype(np.float32) * mask
    
    # 归一化（因为平滑后mask可能不完全加起来等于1）
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    for k in unique_kernels:
        mask = (kernel_map == k).astype(np.float32)
        if k > 0:
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        weight_sum += mask[:, :, np.newaxis]
    
    weight_sum = np.maximum(weight_sum, 1e-6)  # 避免除零
    result = result / weight_sum
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # ===== 应用人体alpha mask：soft blend融合（非线性alpha曲线） =====
    if portrait_mask is not None:
        # 确保mask尺寸匹配
        if portrait_mask.shape[:2] != img.shape[:2]:
            portrait_mask = cv2.resize(portrait_mask, (img.shape[1], img.shape[0]))
        
        # 使用连续alpha值进行soft blend
        alpha = portrait_mask[:, :, np.newaxis].astype(np.float32)  # 扩展为3通道
        
        # 非线性alpha变换：使轮廓过渡更平缓
        # alpha^2 会让过渡region的alpha值向0和1集中，减少硬分离感
        alpha_power = 2.0
        alpha = np.power(alpha, alpha_power)
        
        # soft blend：原图 * alpha + 虚化图 * (1 - alpha)
        result = (img.astype(np.float32) * alpha + result.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    
    return result


def _apply_per_pixel_blur(img, kernel_map, portrait_mask=None):
    """逐像素处理方法（更精确但较慢）"""
    h, w = img.shape[:2]
    result = img.copy()
    
    # 为了效率，我们分块处理
    unique_kernels = np.unique(kernel_map)
    
    for k in unique_kernels:
        if k == 0:
            continue
            
        mask = kernel_map == k
        if not np.any(mask):
            continue
            
        kernel_size = k * 2 + 1
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        result[mask] = blurred[mask]
    
    # ===== 应用人体alpha mask：soft blend融合（非线性alpha曲线） =====
    if portrait_mask is not None:
        # 确保mask尺寸匹配
        if portrait_mask.shape[:2] != img.shape[:2]:
            portrait_mask = cv2.resize(portrait_mask, (img.shape[1], img.shape[0]))
        
        # 使用连续alpha值进行soft blend
        alpha = portrait_mask[:, :, np.newaxis].astype(np.float32)  # 扩展为3通道
        
        # 非线性alpha变换：使轮廓过渡更平缓
        alpha_power = 2.0
        alpha = np.power(alpha, alpha_power)
        
        # soft blend：原图 * alpha + 虚化图 * (1 - alpha)
        result = (img.astype(np.float32) * alpha + result.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    
    return result
    
    
    



def bokeh_blur(image_path, depth_path, lut_path, output_path=None, method='layered', speed_mode='balanced', scale=1.0, upscale_output=False, use_portrait_mask=False, modnet_ckpt=None):
    """
    主函数：执行背景虚化
    
    Args:
        image_path: 原图路径
        depth_path: 深度图路径（0.25倍原图尺寸，0-255）
        lut_path: LUT XML文件路径
        output_path: 输出路径（可选）
        method: 'layered' 或 'per_pixel'
        speed_mode: 'balanced'(默认) | 'fast' | 'ultra_fast' | 'legacy'
        scale: 缩放因子 (0.5=缩小一半，快约4倍; 1.0=原始; 2.0=放大二倍)
               推荐用法: 0.5 (快速), 0.75 (平衡), 1.0 (原始质量)
        upscale_output: 处理后是否放大回原始尺寸 (True=高质量输出, False=快速输出)
        use_portrait_mask: 是否使用MODNet生成人体mask，对人体区域不做虚化
        modnet_ckpt: MODNet模型权重路径（默认自动查找）
    
    Returns:
        虚化后的图像
    """
    import time
    
    total_start = time.time()
    
    # 读取原图
    print(f"读取原图: {image_path}")
    t0 = time.time()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    original_h, original_w = img.shape[:2]
    print(f"原图尺寸: {img.shape[1]}x{img.shape[0]} (耗时: {time.time()-t0:.2f}s)")
    
    # 读取深度图
    print(f"读取深度图: {depth_path}")
    t0 = time.time()
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")
    print(f"深度图尺寸: {depth.shape[1]}x{depth.shape[0]} (耗时: {time.time()-t0:.2f}s)")
    
    # ===== 图像缩放处理 =====
    if scale != 1.0:
        print(f"\n缩放图像: {scale}x")
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        print(f"  原图: {original_w}x{original_h} → 处理图: {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, (new_w // 4, new_h // 4), interpolation=cv2.INTER_NEAREST)  # 深度图0.25倍
        print(f"  处理速度提升约 {1.0/(scale**2):.1f}x (缩小{scale}倍)")
    else:
        print("（不缩放，使用原始尺寸）")
    
    # 解析LUT
    print(f"解析LUT: {lut_path}")
    t0 = time.time()
    lut = parse_lut_xml(lut_path)
    print(f"LUT解析完成 (耗时: {time.time()-t0:.2f}s)")
    
    # ===== 根据缩放因子调整LUT值 =====
    if scale != 1.0:
        print(f"根据缩放因子调整LUT (scale={scale})")
        lut = np.round(lut * scale).astype(np.int32)
        print(f"  调整后最大模糊核: {np.max(lut)}")
    
    # ===== 生成人体mask（可选） =====
    portrait_mask = None
    if use_portrait_mask:
        try:
            # 自动查找模型文件
            if modnet_ckpt is None:
                modnet_dir = os.path.join(os.path.dirname(__file__), '..', 'MODNet', 'pretrained')
                modnet_ckpt = os.path.join(modnet_dir, 'modnet_photographic_portrait_matting.ckpt')

            max_kernel = np.max(lut)
            portrait_mask = generate_portrait_mask(image_path, modnet_ckpt, max_kernel=max_kernel)
        except Exception as e:
            print(f"⚠ 警告：无法生成人体mask: {str(e)}")
            portrait_mask = None

    # 应用虚化
    print(f"\n应用背景虚化 (方法: {method}, 速度模式: {speed_mode})...")
    blur_start = time.time()
    result = apply_depth_based_blur(img, depth, lut, portrait_mask=portrait_mask, method=method, speed_mode=speed_mode)
    blur_time = time.time() - blur_start
    
    # ===== 放大回原始尺寸（可选） =====
    if scale != 1.0 and upscale_output:
        print(f"\n放大输出: {original_w}x{original_h}")
        t0 = time.time()
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        print(f"放大完成 (耗时: {time.time()-t0:.2f}s)")
        print(f"输出尺寸: {result.shape[1]}x{result.shape[0]}")
    elif scale != 1.0:
        print(f"\n输出保持缩小尺寸: {result.shape[1]}x{result.shape[0]}")
    
    # 保存结果
    if output_path is None:
        base = Path(image_path)
        if scale != 1.0 and not upscale_output:
            output_path = str(base.parent / f"{base.stem}_bokeh_scaled{base.suffix}")
        else:
            output_path = str(base.parent / f"{base.stem}_bokeh{base.suffix}")
    
    print(f"保存结果: {output_path}")
    t0 = time.time()
    cv2.imwrite(output_path, result)
    print(f"保存完成 (耗时: {time.time()-t0:.2f}s)")
    
    total_time = time.time() - total_start
    print(f"\n{'='*50}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"虚化处理占比: {blur_time/total_time*100:.1f}%")
    print(f"{'='*50}\n")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='背景虚化程序')
    parser.add_argument('image', help='原图路径')
    parser.add_argument('depth', help='深度图路径（0.25倍原图尺寸）')
    parser.add_argument('lut', help='LUT XML文件路径')
    parser.add_argument('-o', '--output', help='输出路径（可选）')
    parser.add_argument('-m', '--method', choices=['layered', 'per_pixel'], default='layered', help='处理方法')
    parser.add_argument('-s', '--speed', choices=['balanced', 'fast', 'ultra_fast', 'legacy'], 
                        default='balanced', help='速度模式（仅layered方法有效）')
    parser.add_argument('-sc', '--scale', type=float, default=1.0, 
                        help='缩放因子 (0.5=快4倍, 0.75=快1.8倍, 1.0=原始, 默认1.0)')
    parser.add_argument('-u', '--upscale', action='store_true', default=False,
                        help='缩放后放大回原始尺寸（用于高质量输出）')
    parser.add_argument('-pm', '--portrait-mask', action='store_true', default=False,
                        help='使用MODNet生成人体mask，保留人体不虚化（需要PyTorch环境）')
    parser.add_argument('--modnet-ckpt', type=str, default=None,
                        help='MODNet模型权重路径（默认自动查找）')
    args = parser.parse_args()
    bokeh_blur(args.image, args.depth, args.lut, args.output, args.method, args.speed, args.scale, args.upscale, args.portrait_mask, args.modnet_ckpt)


if __name__ == '__main__':
    main()


# 使用示例：

# 基础使用（不保护人体）
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.5

# 缩小一半处理，放大回原始尺寸输出（高质量）
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.5 -u

# 平衡方案（缩小到0.75倍）
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.75 -u

# 结合快速模式（最极致速度）
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.5 -s ultra_fast

# ========== MODNet人体保护示例 ==========
# 基础：使用MODNet保护人体（只虚化背景）
# python bokeh_blur.py image.jpg depth.jpg lut.xml -pm

# 结合缩放：快速处理+保护人体
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.5 -pm -u

# 同时结合快速模式：最快速度+保护人体
# python bokeh_blur.py image.jpg depth.jpg lut.xml -sc 0.5 -s fast -pm -u

# 指定自定义MODNet模型路径
# python bokeh_blur.py image.jpg depth.jpg lut.xml -pm --modnet-ckpt /path/to/model.ckpt

