#!/usr/bin/env python3
"""
交互式虚化生成器
功能：
1. GUI点击选择对焦点
2. 自动计算ROI深度均值
3. 生成FNO1.0-FNO16.0的LUT
4. 逐个生成虚化图像
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加blur_lut_generator路径（相对路径）
script_dir = Path(__file__).parent
project_root = script_dir.parent
blur_lut_generator_dir = project_root / 'blur_lut_generator'
sys.path.insert(0, str(blur_lut_generator_dir))

from blur_lut_generator import generate_blur_lut, export_lut_xml
from bokeh_blur import bokeh_blur


class InteractiveBokehGenerator:
    def __init__(self, image_path, depth_path, output_dir=None, lut_dir=None, html_dir=None, roi_size=31, use_portrait_mask=False, modnet_ckpt=None):
        """
        初始化交互式虚化生成器
        
        Args:
            image_path: RGB图像路径
            depth_path: 深度图路径
            output_dir: 输出目录，默认为./bokeh_results
            lut_dir: LUT输出目录，默认为../LUT
            html_dir: HTML输出目录，默认为../SUMMARY_HTML
            roi_size: ROI大小，默认31x31（可配置）
            use_portrait_mask: 是否使用MODNet生成人体mask（需要PyTorch环境）
            modnet_ckpt: MODNet模型权重路径（默认自动查找）
        """
        self.image_path = Path(image_path)
        self.depth_path = Path(depth_path)
        self.roi_size = roi_size
        self.use_portrait_mask = use_portrait_mask
        self.modnet_ckpt = modnet_ckpt
        
        # 创建输出目录
        if output_dir is None:
            output_dir = self.image_path.parent / 'bokeh_results'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建LUT输出目录（与IMGS同目录的LUT文件夹）
        if lut_dir is None:
            lut_dir = self.image_path.parent.parent / 'LUT'
        self.lut_dir = Path(lut_dir)
        self.lut_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建HTML输出目录（与IMGS同目录的SUMMARY_HTML文件夹）
        if html_dir is None:
            html_dir = self.image_path.parent.parent / 'SUMMARY_HTML'
        self.html_dir = Path(html_dir)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取图像和深度图
        print(f"Loading image: {self.image_path}")
        self.img = cv2.imread(str(self.image_path))
        if self.img is None:
            raise FileNotFoundError(f"Cannot read image: {self.image_path}")
        
        print(f"Loading depth map: {self.depth_path}")
        self.depth = cv2.imread(str(self.depth_path), cv2.IMREAD_GRAYSCALE)
        if self.depth is None:
            raise FileNotFoundError(f"Cannot read depth map: {self.depth_path}")
        
        self.img_rgb = self.img #cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.clicked_point = None
        self.drag_start = None
        self.drag_end = None
        self.selection_rect = None
        self.focus_depth_value = None
        
        # Display window parameters
        self.display_width = 1920
        self.display_height = 1080
        
        # Calculate scaling factor for mouse coordinate mapping
        self.scale_x = self.img.shape[1] / self.display_width
        self.scale_y = self.img.shape[0] / self.display_height
        
        print(f"Image size: {self.img.shape[1]}x{self.img.shape[0]}")
        print(f"Depth map size: {self.depth.shape[1]}x{self.depth.shape[0]}")
        print(f"Display window: {self.display_width}x{self.display_height}")
        print(f"ROI size: {self.roi_size}x{self.roi_size}")
        print(f"Output dir: {self.output_dir}")
        print(f"LUT dir: {self.lut_dir}")
        if self.use_portrait_mask:
            print(f"✓ 人体mask保护已启用（膨胀+高斯平滑边缘处理）")
    
    def _get_mask_path(self):
        """获取mask文件路径（IMGS/modnetportrait/portrait_mask.png）"""
        # 从image_path向上查找IMGS目录
        path_parts = self.image_path.resolve().parts
        for idx, part in enumerate(path_parts):
            if part == 'IMGS':
                imgs_dir = Path(*path_parts[:idx+1])
                return imgs_dir / 'modnetportrait' / 'portrait_mask.png'
        
        # 后备方案：如果找不到IMGS，使用parent.parent
        return self.image_path.parent.parent / 'modnetportrait' / 'portrait_mask.png'
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for rectangular selection"""
        # Map display coordinates to original image coordinates
        img_x = int(x * self.scale_x)
        img_y = int(y * self.scale_y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (img_x, img_y)
            self.drag_end = None
            self.selection_rect = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None and flags & cv2.EVENT_FLAG_LBUTTON:
                self.drag_end = (img_x, img_y)
                # Redraw with current rectangle
                self._redraw_with_selection()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag_start is not None:
                self.drag_end = (img_x, img_y)
                # Finalize selection
                self.selection_rect = self._normalize_rect(self.drag_start, self.drag_end)
                print(f"[OK] Focus region selected: {self.selection_rect}")
    
    def _normalize_rect(self, pt1, pt2):
        """Normalize rectangle coordinates"""
        x1, y1 = pt1
        x2, y2 = pt2
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    
    def _redraw_with_selection(self):
        """Redraw image with current selection rectangle"""
        if hasattr(self, '_window_exists') and self._window_exists:
            # Resize image to display size
            img_display = cv2.resize(self.img_rgb, (self.display_width, self.display_height))
            
            # Draw guidance text
            cv2.putText(img_display, 'Drag to select focus region, then press any key', 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Draw current selection rectangle (scaled to display coordinates)
            if self.drag_start is not None and self.drag_end is not None:
                rect = self._normalize_rect(self.drag_start, self.drag_end)
                # Scale rectangle coordinates to display size
                disp_rect = (int(rect[0] / self.scale_x), int(rect[1] / self.scale_y),
                            int(rect[2] / self.scale_x), int(rect[3] / self.scale_y))
                cv2.rectangle(img_display, (disp_rect[0], disp_rect[1]), 
                             (disp_rect[2], disp_rect[3]), (0, 255, 0), 2)
                # Draw filled semi-transparent rectangle
                overlay = img_display.copy()
                cv2.rectangle(overlay, (disp_rect[0], disp_rect[1]), 
                             (disp_rect[2], disp_rect[3]), (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.2, img_display, 0.8, 0, img_display)
            
            cv2.imshow('Select Focus Region', img_display)
    
    def select_focus_point(self):
        """Interactive focus region selection (rectangular drag)"""
        print("\n" + "="*70)
        print("Drag to select focus region")
        print("Tip: Click and drag to create a rectangle, then press any key to close")
        print("="*70 + "\n")
        
        # Display guidance on image (resized to display size)
        img_display = cv2.resize(self.img_rgb, (self.display_width, self.display_height))
        cv2.putText(img_display, 'Drag to select focus region, then press any key', 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Create window with specified size
        cv2.namedWindow('Select Focus Region', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Select Focus Region', self.display_width, self.display_height)
        
        self._window_exists = True
        cv2.setMouseCallback('Select Focus Region', self.mouse_callback)
        cv2.imshow('Select Focus Region', img_display)
        
        # Wait for selection
        while self.selection_rect is None:
            key = cv2.waitKey(100)
            if key != -1 and key != 255 and self.selection_rect is None:
                print("[Warning] No region selected, please try again")
        
        self._window_exists = False
        cv2.destroyAllWindows()
        return self.selection_rect
    
    def calculate_roi_depth(self, region):
        """
        计算矩形区域的平均深度值
        
        Args:
            region: (x1, y1, x2, y2) 矩形区域坐标（RGB图像坐标）
        
        Returns:
            roi_depth: 0-255的深度值
        """
        x1, y1, x2, y2 = region
        
        # 需要把RGB图像坐标转换到深度图坐标
        # RGB图像尺寸 -> 深度图尺寸的缩放比
        scale_x = self.depth.shape[1] / self.img.shape[1]
        scale_y = self.depth.shape[0] / self.img.shape[0]
        
        # 转换坐标
        x1_depth = int(x1 * scale_x)
        y1_depth = int(y1 * scale_y)
        x2_depth = int(x2 * scale_x)
        y2_depth = int(y2 * scale_y)
        
        # 确保坐标有效
        x1_depth = max(0, x1_depth)
        y1_depth = max(0, y1_depth)
        x2_depth = min(self.depth.shape[1], x2_depth)
        y2_depth = min(self.depth.shape[0], y2_depth)
        
        # 提取ROI并计算平均值
        roi = self.depth[y1_depth:y2_depth, x1_depth:x2_depth]
        
        # Prevent NaN from empty ROI
        if roi.size == 0:
            print(f"[Warning] Empty ROI, using average of entire depth map")
            roi_depth = int(np.mean(self.depth))
        else:
            roi_depth = int(np.mean(roi))
        
        print(f"\nSelected Region Analysis:")
        print(f"  Image Region: ({x1}, {y1}) - ({x2}, {y2})")
        print(f"  Depth Region: ({x1_depth}, {y1_depth}) - ({x2_depth}, {y2_depth})")
        print(f"  Region Size: {roi.shape[1]}x{roi.shape[0]}")
        print(f"  Depth Range: {roi.min()} - {roi.max()}")
        print(f"  Average Depth: {roi_depth}")
        
        return roi_depth
    
    def generate_all_luts(self, focus_depth):
        """
        为所有F数生成LUT
        
        Args:
            focus_depth: 对焦深度值 (0-255)
        
        Returns:
            fno_lut_dict: {fno: (lut_array, xml_path)}
        """
        fno_list = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.2, 16.0]
        fno_lut_dict = {}
        
        print(f"\n{'='*70}")
        print(f"Generating LUTs for focus depth value {focus_depth}")
        print(f"{'='*70}\n")
        
        for fno in fno_list:
            print(f"Generating F/{fno} LUT...", end=' ')
            
            # Generate LUT
            # 焦平面宽度随F数增大而增大：宽度 = focus_width_base + fno * focus_width_factor
            lut = generate_blur_lut(focus_depth, fno, 
                                   focal_length=12,
                                   sensor_width=7.0,
                                   image_width=4096,
                                   max_blur=27,
                                   min_dist=0.5,
                                   max_dist=20.0,
                                   focus_width_base=80,    # 基础焦平面宽度（深度索引范围）
                                   focus_width_factor=0.5) # 焦平面宽度随F数的增长因子
            
            # Save LUT as XML to LUT directory
            lut_filename = self.lut_dir / f'lut_focus_fno{fno:.1f}.xml'
            export_lut_xml(lut, str(lut_filename))
            
            fno_lut_dict[fno] = (lut, lut_filename)
            
            max_blur = np.max(lut)
            print(f"[OK] (Max blur: {max_blur}px)")
        
        return fno_lut_dict
    
    def generate_portrait_mask_once(self):
        """
        生成一次人体mask并保存到磁盘
        后续所有虚化处理都使用这个预生成的mask，避免重复计算
        
        Returns:
            portrait_mask_path: 保存的mask文件路径（如果禁用mask则返回None）
        """
        if not self.use_portrait_mask:
            return None
        
        # 获取正确的mask路径（IMGS/modnetportrait/portrait_mask.png）
        # 不在这里创建目录，由 generate_portrait_mask() 自动处理
        portrait_mask_path = self._get_mask_path()
        
        # 如果mask已存在，检查是否有效
        if portrait_mask_path.exists():
            # 读取缓存的mask，验证是否为连续alpha值（非二值化）
            import cv2
            cached_mask = cv2.imread(str(portrait_mask_path), cv2.IMREAD_GRAYSCALE)
            if cached_mask is not None:
                # 检查是否为有效的连续alpha值（应该有多个唯一值）
                unique_values = np.unique(cached_mask)
                is_continuous = len(unique_values) > 10  # 连续值应该有很多唯一的灰度值
                is_not_empty = cached_mask.max() > 0      # 不是全黑
                
                if is_continuous and is_not_empty:
                    print(f"[OK] 使用已存在的人体alpha mask: {portrait_mask_path}")
                    print(f"     唯一值个数: {len(unique_values)} (连续alpha)")
                    return portrait_mask_path
                else:
                    reason = "全黑" if not is_not_empty else "二值化或无效"
                    print(f"[WARNING] 缓存的mask无效（{reason}），将重新生成")
                    portrait_mask_path.unlink()  # 删除无效的mask
        
        print(f"\n{'='*70}")
        print(f"生成人体mask (第一次，后续将复用)")
        print(f"{'='*70}\n")
        
        try:
            from bokeh_blur import generate_portrait_mask
            
            # 自动查找模型文件
            if self.modnet_ckpt is None:
                modnet_dir = Path(__file__).parent.parent / 'MODNet' / 'pretrained'
                modnet_ckpt = str(modnet_dir / 'modnet_photographic_portrait_matting.ckpt')
            else:
                modnet_ckpt = self.modnet_ckpt
            
            print(f"正在加载MODNet模型...")
            print(f"输入图片: {self.image_path}")
            print(f"生成mask到: IMGS/modnetportrait/portrait_mask.png")
            
            # generate_portrait_mask会自动保存PNG文件到IMGS/modnetportrait/portrait_mask.png
            portrait_mask = generate_portrait_mask(str(self.image_path), modnet_ckpt)
            
            print(f"generate_portrait_mask返回成功，返回值类型: {type(portrait_mask)}")
            
            # PNG已经由generate_portrait_mask直接保存，获取保存路径
            portrait_mask_path = self._get_mask_path()
            
            # 验证文件是否创建
            import time
            time.sleep(0.5)  # 等待文件系统同步
            if portrait_mask_path.exists():
                file_size = portrait_mask_path.stat().st_size
                print(f"✅ 文件验证成功: {portrait_mask_path} (大小: {file_size} bytes)")
            else:
                print(f"⚠ 警告: 文件未找到: {portrait_mask_path}")
            
            return portrait_mask_path
            
        except Exception as e:
            print(f"[ERROR] 无法生成人体mask: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_bokeh_images(self, fno_lut_dict):
        """
        使用生成的LUT逐个生成虚化图像
        
        Args:
            fno_lut_dict: {fno: (lut_array, xml_path)}
        
        Returns:
            results: {fno: output_path}
        """
        results = {}
        total = len(fno_lut_dict)
        
        print(f"\n{'='*70}")
        print(f"Generating bokeh images ({total} images)")
        print(f"{'='*70}\n")
        
        # 先生成一次人体mask（如果启用的话），后续复用
        portrait_mask_path = self.generate_portrait_mask_once()
        
        for idx, (fno, (lut, lut_path)) in enumerate(fno_lut_dict.items(), 1):
            output_filename = self.output_dir / f'bokeh_fno{fno:.1f}.jpg'
            
            print(f"[{idx}/{total}] Processing F/{fno}...", end=' ')
            
            try:
                bokeh_blur(str(self.image_path), 
                          str(self.depth_path),
                          str(lut_path),
                          str(output_filename),
                          method='layered',
                          speed_mode='balanced',
                          scale=1.0,
                          upscale_output=False,
                          use_portrait_mask=self.use_portrait_mask,
                          modnet_ckpt=self.modnet_ckpt)
                results[fno] = output_filename
                print(f"[OK]")
            except Exception as e:
                print(f"[ERROR] {e}")
        
        return results
    
    def visualize_results_summary(self, fno_lut_dict, results):
        """Generate result summary visualization"""
        fno_list = sorted(fno_lut_dict.keys())
        lut_data = [fno_lut_dict[fno][0] for fno in fno_list]
        
        # Draw LUT comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: LUT curve comparison
        colors = plt.cm.viridis(np.linspace(0, 1, len(fno_list)))
        for idx, fno in enumerate(fno_list):
            axes[0].plot(lut_data[idx], linewidth=2, label=f'F/{fno}', color=colors[idx])
        
        axes[0].set_xlabel('Depth Index (0=far, 255=near)', fontsize=11)
        axes[0].set_ylabel('Blur Kernel Radius (pixels)', fontsize=11)
        axes[0].set_title('LUT Comparison for Different F-numbers', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper left', fontsize=9)
        
        # Right: Maximum blur value comparison
        max_blurs = [np.max(lut) for lut in lut_data]
        axes[1].bar(range(len(fno_list)), max_blurs, color=colors, edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(len(fno_list)))
        axes[1].set_xticklabels([f'F/{fno}' for fno in fno_list], rotation=45)
        axes[1].set_ylabel('Max Blur Kernel (pixels)', fontsize=11)
        axes[1].set_title('Maximum Blur Value Comparison', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(max_blurs):
            axes[1].text(i, v + 0.5, str(int(v)), ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        summary_path = self.output_dir / 'lut_summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"\n[OK] Generated LUT summary chart: {summary_path}")
        plt.close()
    
    def generate_web_gallery(self, results):
        """Generate HTML preview page"""
        html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>虚化效果预览</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .info-box {
            background: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .gallery-item {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            transition: transform 0.3s;
        }
        .gallery-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }
        .gallery-item img {
            width: 100%;
            height: auto;
            display: block;
        }
        .gallery-item-title {
            padding: 15px;
            background: #f9f9f9;
            font-weight: bold;
            text-align: center;
            color: #333;
        }
    </style>
</head>
<body>
    <h1>🎬 背景虚化效果预览</h1>
    <div class="info-box">
        <p><strong>生成时间:</strong> """ + str(Path.cwd()) + """</p>
        <p><strong>F数范围:</strong> F/1.0 - F/16.0</p>
        <p><strong>总共虚化图像:</strong> """ + str(len(results)) + """ 张</p>
    </div>
    <div class="gallery">
"""
        
        for fno in sorted(results.keys()):
            img_path = Path(results[fno])
            # 计算从html_dir到图片的相对路径
            try:
                # 转换为绝对路径确保跨卷符工作
                img_abs = img_path.resolve()
                html_abs = self.html_dir.resolve()
                img_rel_path = os.path.relpath(img_abs, html_abs)
                # 转换为前向斜杠（HTML中使用）
                img_rel_path = img_rel_path.replace('\\', '/')
            except (ValueError, OSError):
                # 不同卷符或其他错误，使用文件名（bokeh_results在同一父目录）
                img_rel_path = f"../IMGS/bokeh_results/{img_path.name}"
            
            html_content += f"""        <div class="gallery-item">
            <img src="{img_rel_path}" alt="F/{fno}">
            <div class="gallery-item-title">F/{fno}</div>
        </div>
"""
        
        # 添加原始图片在最后
        try:
            original_image_rel_path = os.path.relpath(self.image_path.resolve(), self.html_dir.resolve())
            original_image_rel_path = original_image_rel_path.replace('\\', '/')
        except (ValueError, OSError):
            original_image_rel_path = "../IMGS/src.jpg"
        
        html_content += f"""        <div class="gallery-item">
            <img src="{original_image_rel_path}" alt="原始照片">
            <div class="gallery-item-title">原始照片</div>
        </div>
"""
        
        html_content += """    </div>
</body>
</html>
"""
        
        html_path = self.html_dir / 'preview.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[OK] Generated HTML preview: {html_path}")
    
    def run(self):
        """Run complete bokeh generation pipeline"""
        try:
            # 1. Interactive focus point selection
            focus_point = self.select_focus_point()
            
            # 2. Calculate ROI average depth
            focus_depth = self.calculate_roi_depth(focus_point)
            self.focus_depth_value = focus_depth
            
            # 3. Generate all LUTs
            fno_lut_dict = self.generate_all_luts(focus_depth)
            
            # 4. Generate bokeh images
            results = self.generate_bokeh_images(fno_lut_dict)
            
            # 5. Generate result summary
            self.visualize_results_summary(fno_lut_dict, results)
            
            # 6. Generate HTML preview
            self.generate_web_gallery(results)
            
            # Print final summary
            print(f"\n{'='*70}")
            print(f"[SUCCESS] Bokeh generation completed!")
            print(f"{'='*70}")
            print(f"Focus point: {focus_point}")
            print(f"Focus depth value: {focus_depth}")
            print(f"Generated images: {len(results)} images")
            print(f"Output directory: {self.output_dir}")
            print(f"LUT directory: {self.lut_dir}")
            print(f"{'='*70}\n")
            
            return results
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    # Get script directory as base path
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    # Configuration parameters (relative paths)
    image_path = base_dir / 'IMGS' / 'src' / '0.jpg'
    depth_path = base_dir / 'IMGS' / 'DEPTH' / '0.png'
    output_dir = base_dir / 'IMGS' / 'bokeh_results'
    lut_dir = base_dir / 'LUT'
    html_dir = base_dir / 'SUMMARY_HTML'
    roi_size = 32  # Configurable ROI size
    
    # MODNet人体保护功能配置
    use_portrait_mask = True  # 设置为 True 启用人体mask保护，False 禁用
    modnet_ckpt = None  # 如果为 None，则自动查找模型文件
    
    print("\n" + "="*70)
    print("🎬 Interactive Bokeh Generator v1.0")
    print("="*70)
    print(f"RGB image: {image_path}")
    print(f"Depth map:  {depth_path}")
    print(f"Output dir: {output_dir}")
    print(f"LUT dir: {lut_dir}")
    print(f"HTML dir: {html_dir}")
    print(f"ROI size: {roi_size}x{roi_size}")
    if use_portrait_mask:
        print(f"✓ 人体mask保护: 已启用")
    else:
        print(f"✗ 人体mask保护: 已禁用")
    print("="*70 + "\n")
    
    # Create generator
    try:
        generator = InteractiveBokehGenerator(
            image_path=str(image_path),
            depth_path=str(depth_path),
            output_dir=str(output_dir),
            lut_dir=str(lut_dir),
            html_dir=str(html_dir),
            roi_size=roi_size,
            use_portrait_mask=use_portrait_mask,
            modnet_ckpt=modnet_ckpt
        )
        
        # Run generation pipeline
        generator.run()
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

