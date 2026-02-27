import numpy as np
import matplotlib.pyplot as plt


def depth_index_to_distance(index, min_dist=0.3, max_dist=10.0):
    """
    将深度索引(0-255)转换为假设的物理距离(米)
    
    ⚠️ 注意：对于单目AI深度估计
    - 深度值0-255是相对深度，不是真实距离
    - 这里的距离是"假设值"，用于应用光学公式
    - min_dist和max_dist可以根据实际场景调整
    - 即使假设不完全准确，相对关系仍然有效
    
    参数:
        index: int, 深度索引 (0=最远, 255=最近)
        min_dist: float, 假设的最近距离(米)，默认0.3m
        max_dist: float, 假设的最远距离(米)，默认10m
    
    返回:
        float: 假设的物理距离(米)
    
    推荐设置：
        - 室内场景: min_dist=0.3, max_dist=5.0
        - 室外场景: min_dist=0.5, max_dist=20.0
        - 人像场景: min_dist=0.5, max_dist=8.0
    """
    # 使用指数映射，让近处（深度值大）的分辨率更高
    # 这符合深度估计模型的特性：近处更准确
    t = index / 255.0
    # 反向指数映射（0->远，255->近）
    distance = max_dist * np.exp(-t * np.log(max_dist / min_dist))
    return distance


def calculate_coc(object_dist, focus_dist, focal_length, fno, sensor_width=6.17):
    """
    计算弥散圆直径 (Circle of Confusion) - 基于真实光学公式
    
    CoC公式推导自薄透镜成像:
    CoC = |A * (v' - v) / v'|
    其中: A = f/N (光圈直径)
          v = 对焦物体的像距
          v' = 实际物体的像距
    
    参数:
        object_dist: float, 物体距离(米)
        focus_dist: float, 对焦距离(米)
        focal_length: float, 焦距(毫米)
        fno: float, 光圈F数
        sensor_width: float, 传感器宽度(毫米)，默认1/2.55英寸
    
    返回:
        float: 弥散圆直径(毫米)
    """
    # 光圈直径 = 焦距 / F数
    aperture = focal_length / fno
    
    # 焦距转换为米
    f = focal_length / 1000.0
    
    # 防止除零和无效值
    if object_dist <= f or focus_dist <= f:
        return 0.0
    
    # 薄透镜成像公式: 1/f = 1/u + 1/v, 得到 v = (f*u)/(u-f)
    focus_image_dist = (f * focus_dist) / (focus_dist - f)
    object_image_dist = (f * object_dist) / (object_dist - f)
    
    # 弥散圆直径计算 (毫米)
    coc = abs(aperture * (object_image_dist - focus_image_dist) / object_image_dist)
    
    return coc


def coc_to_blur_kernel(coc, image_width=4096, sensor_width=6.17):
    """
    将弥散圆直径转换为图像空间的模糊核大小(像素)
    
    参数:
        coc: float, 弥散圆直径(毫米)
        image_width: int, 图像宽度(像素)
        sensor_width: float, 传感器宽度(毫米)
    
    返回:
        float: 模糊核半径(像素)
    """
    # 每个像素对应的传感器尺寸
    pixel_size = sensor_width / image_width
    
    # 模糊核大小(直径) = CoC / 像素大小
    kernel_size = coc / pixel_size
    
    # 返回半径而不是直径，更符合高通的实现
    return kernel_size / 2.0


def generate_blur_lut(focus_index, fno, focal_length=23.0, sensor_width=6.5, 
                      image_width=4096, max_blur=32, min_dist=0.5, max_dist=20.0,
                      focus_width_base=5, focus_width_factor=0.5):
    """
    生成基于光学模型的模糊核查找表 - 适配单目AI深度估计
    
    ⚠️ 重要说明 - 针对单目深度估计：
    虽然单目深度估计只提供0-255的相对深度值（不是真实距离），
    但我们通过假设一个合理的深度范围，仍然可以应用物理光学模型。
    
    为什么这样做有效？
    1. 深度估计的相对关系是准确的（近的确实近，远的确实远）
    2. 光学公式保证了模糊随距离的合理变化规律
    3. F数对模糊的影响符合物理规律（F数减半，模糊倍增）
    4. 即使假设距离不完全准确，生成的LUT仍然比纯数学拟合更自然
    
    如何调整以匹配实际场景？
    - 如果虚化过强 → 增大max_dist（假设远处更远）
    - 如果虚化过弱 → 减小max_dist（假设远处更近）
    - 如果近景虚化不够 → 减小min_dist
    - 如果远景虚化不够 → 增大max_dist
    
    参数:
        focus_index: int, 对焦位置的深度索引 (0-255)
        fno: float, 光圈F数 (1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.2, 16.0)
        focal_length: float, 等效焦距(毫米)，默认23mm
        sensor_width: float, 传感器宽度(毫米)，默认6.5mm
        image_width: int, 图像宽度(像素)，默认4096
        max_blur: int, 最大模糊核半径限制，默认32
        min_dist: float, 假设的最近距离(米)，默认0.5m
        max_dist: float, 假设的最远距离(米)，默认20m
        focus_width_base: float, 焦平面宽度基准值(深度索引范围)，默认5
                         控制焦点区域在F1.0时的宽度
        focus_width_factor: float, 焦平面宽度随F数的增长因子，默认0.5
                           焦平面宽度 = focus_width_base + fno * focus_width_factor
                           例如：F1.0时宽度=5.5, F2.8时宽度=6.9, F16.0时宽度=13
    
    返回:
        numpy.array: 256个元素的模糊核半径数组
        
    场景推荐设置:
        室内场景: min_dist=0.3, max_dist=5.0
        室外场景: min_dist=0.5, max_dist=20.0
        人像场景: min_dist=0.5, max_dist=8.0
    """
    lut = np.zeros(256, dtype=float)
    
    # ===== 计算焦平面宽度 =====
    focus_width = focus_width_base + fno * focus_width_factor
    focus_smooth_width = int(focus_width)
    
    print(f"  焦平面宽度参数: base={focus_width_base}, factor={focus_width_factor}")
    print(f"  F/{fno} 的焦平面宽度: ±{focus_smooth_width} (深度索引范围)")
    lut = np.zeros(256, dtype=float)
    
    # 计算对焦距离（基于假设范围）
    focus_distance = depth_index_to_distance(focus_index, min_dist, max_dist)
    
    for i in range(256):
        # 235-255: 最近距离区域，超出最小对焦距离，不模糊
        if i >= 235:
            lut[i] = 0
        else:
            # 计算物体距离（基于假设范围）
            object_distance = depth_index_to_distance(i, min_dist, max_dist)
            
            # 计算弥散圆直径
            coc = calculate_coc(object_distance, focus_distance, focal_length, fno, sensor_width)
            
            # 转换为像素空间的模糊核半径
            blur_radius = coc_to_blur_kernel(coc, image_width, sensor_width)
            
            # 前景模糊增强（真实相机前景散景通常更强）
            if i > focus_index:  # 前景（index大=距离近）
                blur_radius *= 1.15
            
            # 限制最大模糊
            blur_radius = min(blur_radius, max_blur)
            
            lut[i] = blur_radius
    
    # 近距离平滑过渡 (230-235)
    transition_start = 228
    for i in range(transition_start, 235):
        t = (235 - i) / (235 - transition_start)
        # 使用平滑的余弦函数
        smooth_factor = (1 + np.cos((1 - t) * np.pi)) / 2
        lut[i] = lut[i] * smooth_factor
    
    # 对焦区域高斯平滑，避免完美对焦点的不自然
    # focus_smooth_width 已在上面根据F数动态计算
    for i in range(max(0, focus_index - focus_smooth_width), 
                   min(256, focus_index + focus_smooth_width + 1)):
        dist = abs(i - focus_index)
        if dist <= focus_smooth_width:
            # 高斯权重
            gaussian = np.exp(-(dist ** 2) / (2 * (focus_smooth_width / 2.5) ** 2))
            # 在对焦点附近轻微降低模糊，但不完全清零
            lut[i] = lut[i] * (1 - gaussian * 0.85)
    
    # 取整并转换为整数
    return np.round(lut).astype(int)


def visualize_multiple_fno(focus_index, fno_list):
    """可视化多个F数的LUT对比 - 基于真实光学模型"""
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(fno_list)))
    
    for idx, fno in enumerate(fno_list):
        lut = generate_blur_lut(focus_index, fno)
        plt.plot(lut, linewidth=2, label=f'F/{fno}', color=colors[idx], alpha=0.8)
    
    plt.axvline(x=focus_index, color='red', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'对焦位置: {focus_index}')
    plt.axvline(x=235, color='orange', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='近距离边界: 235')
    
    plt.xlabel('深度索引 (0=远 10m, 255=近 0.3m)', fontsize=12)
    plt.ylabel('模糊核半径 (像素)', fontsize=12)
    plt.title('不同光圈值的模糊核LUT - 基于真实光学模型 (CoC公式)', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(loc='upper left', fontsize=10, framealpha=0.9)
    plt.ylim(-1, 35)
    
    # 添加说明文本
    info_text = f'等效焦距: 26mm | 传感器: 1/2.55" (6.17mm) | 图像: 4000px'
    plt.text(0.98, 0.02, info_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def visualize_single_lut(lut, focus_index, fno):
    """可视化单个LUT - 基于真实光学模型"""
    plt.figure(figsize=(12, 6))
    plt.plot(lut, linewidth=2.5, color='#2E86AB', alpha=0.8)
    plt.fill_between(range(256), lut, alpha=0.2, color='#2E86AB')
    
    plt.axvline(x=focus_index, color='red', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'对焦位置: {focus_index}')
    plt.axvline(x=235, color='green', linestyle='--', 
                linewidth=2, alpha=0.7, label='近距离边界: 235')
    
    plt.xlabel('深度索引 (0=远 10m, 255=近 0.3m)', fontsize=11)
    plt.ylabel('模糊核半径 (像素)', fontsize=11)
    plt.title(f'模糊核LUT - F/{fno} (对焦位置={focus_index}) - 真实光学模型', fontsize=13)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.legend(fontsize=10)
    plt.ylim(-1, 35)
    
    # 标注关键信息
    max_blur = np.max(lut)
    max_idx = np.argmax(lut)
    plt.annotate(f'最大模糊: {max_blur}px\n@index {max_idx}', 
                xy=(max_idx, max_blur), xytext=(max_idx + 20, max_blur - 3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=9, color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def export_lut_xml(lut, filename):
    """导出为XML格式"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('<param name = "map_lut" type = "int" format = "1d_array" size = "256">\n')
        f.write('    <value>\n')
        # 每行写入多个值
        values_str = ','.join(str(v) for v in lut) + ','
        f.write(values_str)
        f.write('\n   </value>\n')
        f.write('</param>')
    print(f"LUT已导出至: {filename}")


# 示例使用
if __name__ == "__main__":
    # 标准F数列表 (光圈值)
    fno_list = [1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.2, 16.0]
    
    # 对焦位置 (0-255，0=最远，255=最近)
    focus_position = 50  # 对焦在中等深度
    
    # ⚠️ 深度范围设置 - 针对单目AI深度估计
    # 这些是假设值，用于将相对深度映射到物理模型
    # 根据实际场景调整这些值可以优化虚化效果
    min_depth_distance = 0.5   # 假设最近距离(米) - 深度值255对应
    max_depth_distance = 20.0  # 假设最远距离(米) - 深度值0对应
    
    # 场景推荐：
    # 室内场景: min=0.3, max=5.0
    # 室外场景: min=0.5, max=20.0
    # 人像场景: min=0.5, max=8.0
    
    print("="*70)
    print("高通平台模糊核LUT生成器 - 适配单目AI深度估计")
    print("="*70)
    print(f"⚠️  深度类型: 相对深度 (0-255)")
    print(f"    - 深度值 0   = 最远 (假设约 {max_depth_distance}m)")
    print(f"    - 深度值 255 = 最近 (假设约 {min_depth_distance}m)")
    print(f"    - 深度值 {focus_position}  = 对焦点 (假设约 {depth_index_to_distance(focus_position, min_depth_distance, max_depth_distance):.2f}m)")
    print()
    print(f"💡 虚化调整提示:")
    print(f"    - 如果整体虚化过强 → 增大 max_depth_distance")
    print(f"    - 如果整体虚化过弱 → 减小 max_depth_distance")
    print(f"    - 如果近景虚化不够 → 减小 min_depth_distance")
    print()
    print(f"焦平面宽度参数：")
    print(f"    - focus_width_base = 5 (基础宽度)")
    print(f"    - focus_width_factor = 0.5 (F数增长因子)")
    print(f"    - 焦平面宽度 = base + fno * factor")
    print(f"    - 例如：F1.0→5.5, F2.8→6.4, F16.0→13")
    print()
    print(f"光学参数:")
    print(f"  - 等效焦距: 26mm")
    print(f"  - 传感器尺寸: (6.5mm 宽)")
    print(f"  - 图像分辨率: 4096 像素")
    print(f"  - 假设深度范围: {min_depth_distance}m - {max_depth_distance}m")
    print("="*70)
    
    # 生成并分析所有F数的LUT
    lut_data = {}
    for fno in fno_list:
        lut = generate_blur_lut(focus_position, fno, 
                                min_dist=min_depth_distance,
                                max_dist=max_depth_distance,
                                focal_length=23.0,
                                focus_width_base=5,
                                focus_width_factor=0.5)
        lut_data[fno] = lut
        
        print(f"\n{'F/' + str(fno):>6} | 最大模糊: {np.max(lut):3d}px | "
              f"非零元素: {np.count_nonzero(lut):3d} | "
              f"平均模糊: {np.mean(lut[lut > 0]):.1f}px")
        
        # 显示关键位置的值
        print(f"       | 远景(idx=0): {lut[0]:2d}px | "
              f"对焦点(idx={focus_position}): {lut[focus_position]:2d}px | "
              f"前景(idx=150): {lut[150]:2d}px")
        
        # 导出XML
        filename = f'lut_focus_fno{fno:.1f}.xml'
        export_lut_xml(lut, filename)
    
    print("\n" + "="*70)
    print("生成对比分析...")
    print("="*70)
    
    # 物理规律验证
    print("\n物理规律验证 (模糊核与F数应成反比):")
    print(f"{'F数':<8} {'光圈直径(mm)':<15} {'理论模糊倍数':<15} {'实际最大模糊(px)':<15}")
    print("-"*70)
    for fno in [1.0, 2.0, 4.0, 8.0, 16.0]:
        aperture_dia = 8.0 / fno  # 焦距/F数
        theory_ratio = 16.0 / fno  # 相对于F16的理论倍数
        actual_blur = np.max(lut_data[fno])
        print(f"F/{fno:<6} {aperture_dia:<15.2f} {theory_ratio:<15.2f} {actual_blur:<15d}")
    
    # 可视化所有F数的对比
    print("\n生成对比图表...")
    visualize_multiple_fno(focus_position, fno_list)
    
    # 可视化典型F数的单独LUT
    print("\n显示典型F数的详细LUT...")
    for fno in [1.0, 2.8, 5.6, 16.0]:
        lut = lut_data[fno]
        visualize_single_lut(lut, focus_position, fno)
    
    print("\n" + "="*70)
    print("✅ LUT生成完成！")
    print("="*70)