#!/usr/bin/env python3
"""
快速启动脚本 - 交互式虚化生成器
直接运行此脚本即可开始交互式虚化生成流程
"""

from interactive_bokeh_generator import InteractiveBokehGenerator
from pathlib import Path

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
