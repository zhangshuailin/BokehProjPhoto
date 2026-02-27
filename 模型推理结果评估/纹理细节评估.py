import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthEdgeVisualizer:
    def __init__(self, rgb1_path, rgb2_path, depth1_path, depth2_path):
        """
        rgb1_path: 第一个RGB图（对应depth1）
        rgb2_path: 第二个RGB图（对应depth2）
        depth1_path: 第一个深度图（模型A）
        depth2_path: 第二个深度图（模型B）
        """
        # 读取RGB图
        self.rgb1 = cv2.imread(rgb1_path)
        self.rgb1 = cv2.cvtColor(self.rgb1, cv2.COLOR_BGR2RGB)
        
        self.rgb2 = cv2.imread(rgb2_path)
        self.rgb2 = cv2.cvtColor(self.rgb2, cv2.COLOR_BGR2RGB)
        
        # 转换为灰度图（3通道，用于叠加彩色边缘）
        gray1 = cv2.cvtColor(self.rgb1, cv2.COLOR_RGB2GRAY)
        self.rgb1_gray = cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB)
        
        gray2 = cv2.cvtColor(self.rgb2, cv2.COLOR_RGB2GRAY)
        self.rgb2_gray = cv2.cvtColor(gray2, cv2.COLOR_GRAY2RGB)
        
        # 读取深度图
        self.depth1 = cv2.imread(depth1_path, cv2.IMREAD_GRAYSCALE)
        self.depth2 = cv2.imread(depth2_path, cv2.IMREAD_GRAYSCALE)
        
        if self.rgb1 is None or self.rgb2 is None or self.depth1 is None or self.depth2 is None:
            raise ValueError("无法读取图像，请检查路径")
        
        # 确保尺寸一致
        h1, w1 = self.rgb1.shape[:2]
        self.depth1 = cv2.resize(self.depth1, (w1, h1))
        
        h2, w2 = self.rgb2.shape[:2]
        self.depth2 = cv2.resize(self.depth2, (w2, h2))
        
        print(f"图像1尺寸: {self.rgb1.shape[:2]}")
        print(f"图像2尺寸: {self.rgb2.shape[:2]}")
    
    def detect_edges(self, depth_map, low_threshold=30, high_threshold=100, 
                     blur_kernel=5, use_adaptive=False):
        """
        对深度图进行边缘检测
        """
        # 归一化深度图
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(depth_norm, (blur_kernel, blur_kernel), 0)
        
        if use_adaptive:
            median = np.median(blurred)
            low = int(max(0, 0.66 * median))
            high = int(min(255, 1.33 * median))
            edges = cv2.Canny(blurred, low, high)
        else:
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
    def overlay_edges_on_gray(self, gray_rgb, edges, color, edge_thickness=1):
        """
        将边缘叠加到灰度RGB图上，边缘使用指定颜色
        """
        # 复制灰度图
        overlay = gray_rgb.copy()
        
        # 可选：增粗边缘
        if edge_thickness > 1:
            kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 叠加彩色边缘
        overlay[edges > 0] = color
        
        return overlay
    
    def visualize(self, low_threshold=30, high_threshold=100, 
                  edge_thickness=1, use_adaptive=False,
                  color1=(255, 0, 0), color2=(0, 100, 255),
                  save_path=None):
        """
        完整的可视化流程
        
        布局（2行4列）：
        第一行：灰度图1, 深度图1, 边缘叠加图1, 边缘图1
        第二行：灰度图2, 深度图2, 边缘叠加图2, 边缘图2
        """
        print("正在检测边缘...")
        
        # 边缘检测
        edges1 = self.detect_edges(self.depth1, low_threshold, high_threshold, 
                                   use_adaptive=use_adaptive)
        edges2 = self.detect_edges(self.depth2, low_threshold, high_threshold,
                                   use_adaptive=use_adaptive)
        
        # 创建叠加图
        overlay1 = self.overlay_edges_on_gray(self.rgb1_gray, edges1, color1, edge_thickness)
        overlay2 = self.overlay_edges_on_gray(self.rgb2_gray, edges2, color2, edge_thickness)
        
        # 计算边缘统计
        edge_count1 = np.sum(edges1 > 0)
        edge_count2 = np.sum(edges2 > 0)
        
        print(f"\n边缘统计:")
        print(f"深度图1边缘像素: {edge_count1}")
        print(f"深度图2边缘像素: {edge_count2}")
        
        # 可视化
        fig = plt.figure(figsize=(20, 10))
        
        # ========== 第一行：图像1 ==========
        plt.subplot(2, 4, 1)
        plt.imshow(self.rgb1_gray)
        plt.title('灰度图1', fontsize=14, fontproperties='SimHei')
        plt.axis('off')
        
        plt.subplot(2, 4, 2)
        plt.imshow(self.depth1, cmap='jet')
        plt.title('深度图1', fontsize=14, fontproperties='SimHei')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(2, 4, 3)
        plt.imshow(overlay1)
        plt.title(f'边缘叠加图1（红色）\n边缘像素: {edge_count1}', 
                  fontsize=12, fontproperties='SimHei')
        plt.axis('off')
        
        plt.subplot(2, 4, 4)
        plt.imshow(edges1, cmap='gray')
        plt.title('边缘检测图1', fontsize=12, fontproperties='SimHei')
        plt.axis('off')
        
        # ========== 第二行：图像2 ==========
        plt.subplot(2, 4, 5)
        plt.imshow(self.rgb2_gray)
        plt.title('灰度图2', fontsize=14, fontproperties='SimHei')
        plt.axis('off')
        
        plt.subplot(2, 4, 6)
        plt.imshow(self.depth2, cmap='jet')
        plt.title('深度图2', fontsize=14, fontproperties='SimHei')
        plt.axis('off')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        plt.subplot(2, 4, 7)
        plt.imshow(overlay2)
        plt.title(f'边缘叠加图2（蓝色）\n边缘像素: {edge_count2}', 
                  fontsize=12, fontproperties='SimHei')
        plt.axis('off')
        
        plt.subplot(2, 4, 8)
        plt.imshow(edges2, cmap='gray')
        plt.title('边缘检测图2', fontsize=12, fontproperties='SimHei')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n结果已保存到: {save_path}")
        
        plt.show()
        
        return {
            'edges1': edges1,
            'edges2': edges2,
            'overlay1': overlay1,
            'overlay2': overlay2,
            'gray1': self.rgb1_gray,
            'gray2': self.rgb2_gray
        }
    
    def save_individual_results(self, results, prefix="result"):
        """
        单独保存所有结果图，方便详细查看
        """
        # 保存灰度图
        #cv2.imwrite(f"{prefix}_gray1.png", cv2.cvtColor(results['gray1'], cv2.COLOR_RGB2BGR))
        #cv2.imwrite(f"{prefix}_gray2.png", cv2.cvtColor(results['gray2'], cv2.COLOR_RGB2BGR))
        
        # 保存叠加图
        # cv2.imwrite(f"{prefix}_overlay1_red.png", cv2.cvtColor(results['overlay1'], cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"{prefix}_overlay2_blue.png", cv2.cvtColor(results['overlay2'], cv2.COLOR_RGB2BGR))
        
        # 保存边缘图
        # cv2.imwrite(f"{prefix}_edges1.png", results['edges1'])
        # cv2.imwrite(f"{prefix}_edges2.png", results['edges2'])
        
        #print(f"\n单独保存完成:")
        #print(f"  灰度图: {prefix}_gray1.png, {prefix}_gray2.png")
        #print(f"  叠加图: {prefix}_overlay1_red.png, {prefix}_overlay2_blue.png")
        #print(f"  边缘图: {prefix}_edges1.png, {prefix}_edges2.png")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 遍历35张图片，从0到34
    for i in range(35):
        print(f"\n========== 处理图片 {i}.jpg ==========")
        
        # 修改为你的图片路径
        rgb_image1 = f"src/{i}.jpg"        # 第一个RGB图
        rgb_image2 = f"src/{i}.jpg"        # 第二个RGB图
        depth_map1 = f"fastdepth/{i}.png"      # 第一个深度图（对应rgb1）
        depth_map2 = f"DepthAntThingVITS/{i}.png"      # 第二个深度图（对应rgb2）
        
        try:
            # 创建可视化工具
            visualizer = DepthEdgeVisualizer(rgb_image1, rgb_image2, depth_map1, depth_map2)
            
            # 执行可视化
            #results = visualizer.visualize(
            #    low_threshold=5,      # Canny低阈值
            #    high_threshold=21,    # Canny高阈值
            #    edge_thickness=3,      # 边缘粗细
            #    use_adaptive=False,    # 是否自动调整阈值
            #    color1=(255, 0, 0),    # 深度图1边缘颜色（红色）
            #    color2=(0, 0, 255),  # 深度图2边缘颜色（蓝色）
            #    save_path=f"output/depth_edge_comparison_{i}.png"  # 按图片编号保存
            #)
            
            # 可选：单独保存所有图片，方便放大查看
            visualizer.save_individual_results(results, prefix=f"result_{i}")
            
            print(f"✓ 图片 {i} 分析完成!")
            
        except FileNotFoundError as e:
            print(f"✗ 图片 {i} 文件不存在: {e}")
            continue
        except Exception as e:
            print(f"✗ 图片 {i} 处理出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n========== 所有图片处理完成 ==========")
    print(f"共处理了 {i+1} 张图片")
    print("结果保存在 output/ 目录下")