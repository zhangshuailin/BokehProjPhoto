"""
环境检查和验证脚本
用法: python check_environment.py
"""

import sys
import os


def check_python_version():
    """检查Python版本"""
    print('🔍 检查Python版本...')
    version = sys.version_info
    print(f'   Python {version.major}.{version.minor}.{version.micro}', end='')
    if version >= (3, 6):
        print(' ✅')
        return True
    else:
        print(' ❌ (需要 >= 3.6)')
        return False


def check_torch():
    """检查PyTorch"""
    print('🔍 检查PyTorch...')
    try:
        import torch
        print(f'   PyTorch版本: {torch.__version__} ✅')
        
        # 检查CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f'   CUDA可用: 是 ✅')
            print(f'   GPU数量: {torch.cuda.device_count()}')
            print(f'   GPU名称: {torch.cuda.get_device_name(0)}')
        else:
            print(f'   CUDA可用: 否 (将使用CPU，推理较慢)')
        return True
    except ImportError:
        print('   PyTorch未安装 ❌')
        print('   运行: pip install torch torchvision')
        return False


def check_dependencies():
    """检查其他依赖"""
    print('🔍 检查依赖包...')
    
    deps = {
        'torchvision': 'torchvision',
        'PIL': 'pillow',
        'numpy': 'numpy'
    }
    
    all_ok = True
    for module_name, package_name in deps.items():
        try:
            __import__(module_name)
            print(f'   {package_name}: ✅')
        except ImportError:
            print(f'   {package_name}: ❌')
            all_ok = False
    
    return all_ok


def check_model_file():
    """检查模型文件"""
    print('🔍 检查模型文件...')
    model_path = 'pretrained/modnet_photographic_portrait_matting.ckpt'
    
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f'   模型文件: ✅')
        print(f'   路径: {model_path}')
        print(f'   大小: {size_mb:.1f} MB')
        return True
    else:
        print(f'   模型文件: ❌')
        print(f'   预期路径: {model_path}')
        print(f'   请从以下链接下载:')
        print(f'   https://drive.google.com/drive/folders/1umYmlCulvIFNaqPjwod1SayFmSRHziyR?usp=sharing')
        return False


def check_modnet_code():
    """检查MODNet代码"""
    print('🔍 检查MODNet代码...')
    
    files = [
        'src/models/modnet.py',
        'src/models/__init__.py',
        'src/models/backbones/mobilenetv2.py',
    ]
    
    all_ok = True
    for file_path in files:
        if os.path.exists(file_path):
            print(f'   {file_path}: ✅')
        else:
            print(f'   {file_path}: ❌ (缺失)')
            all_ok = False
    
    return all_ok


def test_model_loading():
    """测试模型是否能正常加载"""
    print('🔍 测试模型加载...')
    
    try:
        import torch
        import torch.nn as nn
        from src.models.modnet import MODNet
        
        print('   导入模块: ✅')
        
        # 尝试创建模型
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)
        print('   创建模型实例: ✅')
        
        # 尝试加载权重
        model_path = 'pretrained/modnet_photographic_portrait_matting.ckpt'
        if os.path.exists(model_path):
            weights = torch.load(model_path, map_location='cpu')
            modnet.load_state_dict(weights)
            print('   加载权重: ✅')
            
            modnet.eval()
            print('   模型准备推理: ✅')
            return True
        else:
            print('   模型文件不存在，跳过权重加载')
            return False
            
    except Exception as e:
        print(f'   错误: {str(e)} ❌')
        return False


def main():
    print('\n' + '='*60)
    print('MODNet 环境检查工具')
    print('='*60 + '\n')
    
    checks = [
        ('Python版本', check_python_version),
        ('PyTorch', check_torch),
        ('依赖包', check_dependencies),
        ('MODNet代码', check_modnet_code),
        ('模型文件', check_model_file),
        ('模型加载测试', test_model_loading),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f'   检查失败: {str(e)}')
            results.append((name, False))
        print()
    
    # 总结
    print('='*60)
    print('检查结果总结:')
    print('='*60)
    
    for name, result in results:
        status = '✅' if result else '❌'
        print(f'{status} {name}')
    
    all_passed = all(result for _, result in results)
    
    print('='*60)
    if all_passed:
        print('✅ 所有检查通过！你可以开始使用MODNet推理了。\n')
        print('快速开始:')
        print('  1. 单张图片推理:')
        print('     python simple_inference.py <图片路径>\n')
        print('  2. 批量推理:')
        print('     python run_portrait_matting.py --input-path <文件夹> --output-path <输出文件夹>\n')
    else:
        print('❌ 部分检查未通过，请按照上面的提示修复问题。\n')
    
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
