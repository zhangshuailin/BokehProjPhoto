#!/usr/bin/env python3
"""
快速验证脚本 - 检查环境和依赖
在运行 run_interactive_bokeh.py 之前执行此脚本确保环境正确
"""

import sys
import importlib
from pathlib import Path

# Get project root directory
script_dir = Path(__file__).parent
project_root = script_dir.parent

required_packages = [
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('matplotlib', 'matplotlib'),
]

print("="*70)
print("🔍 环境检查和文件验证")
print("="*70)

print(f"\nPython 版本: {sys.version}")
print(f"Python 位置: {sys.executable}")

# 检查包
print("\n【1. 检查必要的包】")
missing_packages = []

for module_name, package_name in required_packages:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {package_name:<20} v{version}")
    except ImportError:
        print(f"  ✗ {package_name:<20} [未安装]")
        missing_packages.append(package_name)

if missing_packages:
    print(f"\n⚠️  缺少以下包，请安装:")
    install_cmd = ' '.join(missing_packages)
    print(f"  pip install {install_cmd}")
    print(f"\n建议使用:")
    print(f"  pip install -i https://pypi.tsinghua.edu.cn/simple {install_cmd}")
else:
    print("\n✅ 所有必要的包都已安装！")

# 检查文件
print("\n【2. 检查输入文件】")
input_files = [
    (project_root / 'IMGS' / 'src.jpg', 'RGB图像'),
    (project_root / 'DEPTH' / 'depth.png', '深度图'),
]

all_inputs_exist = True
for path, description in input_files:
    if path.exists():
        file_size = path.stat().st_size / (1024*1024)  # MB
        print(f"  ✓ {description:<15} ({file_size:.2f} MB)")
    else:
        print(f"  ✗ {description:<15} [不存在: {path}]")
        all_inputs_exist = False

# 检查依赖脚本
print("\n【3. 检查依赖脚本】")
scripts = [
    (project_root / 'blur_lut_generator' / 'blur_lut_generator.py', 'LUT生成器'),
    (script_dir / 'bokeh_blur.py', '虚化程序'),
    (script_dir / 'interactive_bokeh_generator.py', '交互式生成器'),
]

all_scripts_exist = True
for path, description in scripts:
    if path.exists():
        print(f"  ✓ {description:<15} ({path.stat().st_size} bytes)")
    else:
        print(f"  ✗ {description:<15} [不存在: {path}]")
        all_scripts_exist = False

# 检查输出目录
print("\n【4. 检查输出目录】")
output_dir = project_root / 'IMGS' / 'bokeh_results'
if output_dir.exists():
    print(f"  ✓ 输出目录存在: {output_dir}")
else:
    print(f"  ⚠️  输出目录不存在，将自动创建: {output_dir}")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 已创建输出目录")
    except Exception as e:
        print(f"  ✗ 无法创建输出目录: {e}")

# 总结
print("\n" + "="*70)

success = not missing_packages and all_inputs_exist and all_scripts_exist

if success:
    print("✅ 所有检查通过！可以运行 python run_interactive_bokeh.py")
else:
    print("⚠️  有些项目未通过检查，请根据上面的提示修复")
    if missing_packages:
        print("\n【必须修复】缺少Python包")
    if not all_inputs_exist:
        print("【必须修复】缺少输入文件")
    if not all_scripts_exist:
        print("【必须修复】缺少脚本文件")

print("="*70 + "\n")

sys.exit(0 if success else 1)
