import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import interpolate
import os
from datetime import datetime

class InteractiveDepthProfiler:
    def __init__(self, rgb_path, depth1_path, depth2_path):
        """
        交互式深度剖面查看器
        """
        self.rgb = cv2.imread(rgb_path)
        self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
        
        self.depth1 = self._load_depth(depth1_path)
        self.depth2 = self._load_depth(depth2_path)
        
        h, w = self.rgb.shape[:2]
        self.depth1 = cv2.resize(self.depth1, (w, h))
        self.depth2 = cv2.resize(self.depth2, (w, h))
        
        # 存储用户画的线
        self.lines = []
        self.current_line = []
        
        # 图形对象
        self.fig = None
        self.ax_rgb = None
        self.ax_depth1 = None
        self.ax_depth2 = None
        self.ax_profile1 = None
        self.ax_profile2 = None
        
        # 保存路径
        self.save_dir = "depth_profiles"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"图像尺寸: {self.rgb.shape[:2]}")
        print("\n使用说明:")
        print("="*60)
        print("1. 在RGB图上点击鼠标左键画线")
        print("2. 每条线需要点击2个点（起点和终点）")
        print("3. 画完一条线后，会自动显示深度剖面")
        print("4. 按 'c' 键清除所有线")
        print("5. 按 'u' 键撤销最后一条线")
        print("6. 按 'q' 键或关闭窗口退出并保存")
        print("="*60)
    
    def _load_depth(self, path):
        """加载并归一化深度图"""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
        else:
            depth = depth.astype(np.float32)
        return cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    
    def sample_along_line(self, depth, p1, p2, num_samples=200):
        """
        沿着线段采样深度值
        p1, p2: (x, y) 坐标
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # 生成线段上的采样点
        t = np.linspace(0, 1, num_samples)
        x_coords = x1 + t * (x2 - x1)
        y_coords = y1 + t * (y2 - y1)
        
        # 采样深度值（使用双线性插值）
        depth_values = []
        for x, y in zip(x_coords, y_coords):
            # 边界检查
            x = np.clip(x, 0, depth.shape[1] - 1)
            y = np.clip(y, 0, depth.shape[0] - 1)
            
            # 双线性插值
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, depth.shape[1] - 1), min(y0 + 1, depth.shape[0] - 1)
            
            dx, dy = x - x0, y - y0
            
            depth_val = (1 - dx) * (1 - dy) * depth[y0, x0] + \
                       dx * (1 - dy) * depth[y0, x1] + \
                       (1 - dx) * dy * depth[y1, x0] + \
                       dx * dy * depth[y1, x1]
            
            depth_values.append(depth_val)
        
        return np.array(depth_values), x_coords, y_coords
    
    def on_click(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax_rgb:
            return
        
        if event.button == 1:  # 左键点击
            x, y = int(event.xdata), int(event.ydata)
            self.current_line.append((x, y))
            
            # 在RGB图上标记点击位置
            self.ax_rgb.plot(x, y, 'ro', markersize=8)
            self.fig.canvas.draw()
            
            # 如果已经点击了两个点，画线并显示剖面
            if len(self.current_line) == 2:
                p1, p2 = self.current_line
                self.lines.append((p1, p2))
                
                # 画线
                line_color = plt.cm.rainbow(len(self.lines) / 10)
                self.ax_rgb.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               color=line_color, linewidth=3, alpha=0.7)
                self.ax_depth1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                   color=line_color, linewidth=3, alpha=0.7)
                self.ax_depth2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                   color=line_color, linewidth=3, alpha=0.7)
                
                # 更新剖面图
                self.update_profiles()
                
                # 重置当前线
                self.current_line = []
                
                print(f"✓ 已添加第 {len(self.lines)} 条线: {p1} -> {p2}")
    
    def on_key(self, event):
        """处理键盘事件"""
        if event.key == 'c':  # 清除所有线
            self.lines = []
            self.current_line = []
            self.redraw_all()
            print("已清除所有线")
        
        elif event.key == 'u':  # 撤销最后一条线
            if self.lines:
                self.lines.pop()
                self.redraw_all()
                print(f"已撤销，剩余 {len(self.lines)} 条线")
        
        elif event.key == 'q':  # 退出并保存
            self.save_results()
            plt.close(self.fig)
            print("已保存并退出")
    
    def on_close(self, event):
        """窗口关闭事件"""
        self.save_results()
        print("已自动保存结果")
    
    def save_results(self):
        """保存当前结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存整个figure
        if self.fig:
            fig_path = os.path.join(self.save_dir, f"{name}.png")
            self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"✓ 完整视图已保存: {fig_path}")
            
            # 如果有画线，也保存剖面数据
            #if self.lines:
            #    data_path = os.path.join(self.save_dir, f"profile_data_{timestamp}.txt")
            #    with open(data_path, 'w', encoding='utf-8') as f:
            #        f.write("深度剖面数据\n")
            #        f.write("="*60 + "\n")
            #        f.write(f"保存时间: {datetime.now()}\n")
            #        f.write(f"线条数量: {len(self.lines)}\n\n")
            #        
            #        for i, (p1, p2) in enumerate(self.lines):
            #            depths1, x_coords, y_coords = self.sample_along_line(self.depth1, p1, p2)
            #            depths2, _, _ = self.sample_along_line(self.depth2, p1, p2)
            #            distance = np.sqrt((x_coords - p1[0])**2 + (y_coords - p1[1])**2)
            #            
            #            f.write(f"线 {i+1}: {p1} -> {p2}\n")
            #            f.write(f"  长度: {distance[-1]:.2f} 像素\n")
            #            f.write(f"  DepthAntThingVITS - 最小值: {depths1.min():.4f}, 最大值: {depths1.max():.4f}, 均值: {depths1.mean():.4f}\n")
            #            f.write(f"  fastdepth - 最小值: {depths2.min():.4f}, 最大值: {depths2.max():.4f}, 均值: {depths2.mean():.4f}\n\n")
            #    print(f"✓ 剖面数据已保存: {data_path}")
    
    def update_profiles(self):
        """更新深度剖面图"""
        # 清空剖面图
        self.ax_profile1.clear()
        self.ax_profile2.clear()
        
        # 绘制每条线的剖面
        for i, (p1, p2) in enumerate(self.lines):
            color = plt.cm.rainbow(i / 10)
            
            # 采样深度值
            depths1, x_coords, y_coords = self.sample_along_line(self.depth1, p1, p2)
            depths2, _, _ = self.sample_along_line(self.depth2, p1, p2)
            
            # 计算距离（作为x轴）
            distance = np.sqrt((x_coords - p1[0])**2 + (y_coords - p1[1])**2)
            
            # 绘制剖面
            self.ax_profile1.plot(distance, depths1, color=color, 
                                 linewidth=2, label=f'线 {i+1}', alpha=0.8)
            self.ax_profile2.plot(distance, depths2, color=color,
                                 linewidth=2, label=f'线 {i+1}', alpha=0.8)
        
        # 设置剖面图样式
        self.ax_profile1.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=10)
        self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=10)
        self.ax_profile1.set_title('DepthAntThingVITS - 剖面\n(平滑=连续性好)', 
                                   fontsize=11, fontproperties='SimHei')
        self.ax_profile1.grid(alpha=0.3)
        if self.lines:
            self.ax_profile1.legend(prop={'family': 'SimHei'}, fontsize=9, loc='best')
        
        self.ax_profile2.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=10)
        self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=10)
        self.ax_profile2.set_title('fastdepth - 剖面\n(平滑=连续性好)', 
                                   fontsize=11, fontproperties='SimHei')
        self.ax_profile2.grid(alpha=0.3)
        if self.lines:
            self.ax_profile2.legend(prop={'family': 'SimHei'}, fontsize=9, loc='best')
        
        # 设置相同的y轴范围
        if self.lines:
            y_min = min(self.ax_profile1.get_ylim()[0], self.ax_profile2.get_ylim()[0])
            y_max = max(self.ax_profile1.get_ylim()[1], self.ax_profile2.get_ylim()[1])
            self.ax_profile1.set_ylim(y_min, y_max)
            self.ax_profile2.set_ylim(y_min, y_max)
        
        self.fig.canvas.draw()
    
    def redraw_all(self):
        """重绘所有内容"""
        # 清空所有图
        self.ax_rgb.clear()
        self.ax_depth1.clear()
        self.ax_depth2.clear()
        self.ax_profile1.clear()
        self.ax_profile2.clear()
        
        # 重新显示图像
        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title('RGB图 - 点击画线', 
                             fontsize=12, fontproperties='SimHei')
        self.ax_rgb.axis('off')
        
        self.ax_depth1.imshow(self.depth1, cmap='jet')
        self.ax_depth1.set_title('DepthAntThingVITS', fontsize=12, fontproperties='SimHei')
        self.ax_depth1.axis('off')
        
        self.ax_depth2.imshow(self.depth2, cmap='jet')
        self.ax_depth2.set_title('fastdepth', fontsize=12, fontproperties='SimHei')
        self.ax_depth2.axis('off')
        
        # 重新画线
        for i, (p1, p2) in enumerate(self.lines):
            color = plt.cm.rainbow(i / 10)
            
            self.ax_rgb.plot([p1[0], p2[0]], [p1[1], p2[1]],
                           color=color, linewidth=3, alpha=0.7)
            self.ax_rgb.plot([p1[0], p2[0]], [p1[1], p2[1]], 'ro', markersize=8)
            
            self.ax_depth1.plot([p1[0], p2[0]], [p1[1], p2[1]],
                               color=color, linewidth=3, alpha=0.7)
            self.ax_depth2.plot([p1[0], p2[0]], [p1[1], p2[1]],
                               color=color, linewidth=3, alpha=0.7)
        
        # 更新剖面图
        if self.lines:
            self.update_profiles()
        else:
            self.ax_profile1.text(0.5, 0.5, '请在RGB图上画线', 
                                 ha='center', va='center',
                                 transform=self.ax_profile1.transAxes,
                                 fontproperties='SimHei', fontsize=14)
            self.ax_profile1.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=11)
            self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
            self.ax_profile1.set_title('DepthAntThingVITS - 剖面', fontsize=12, fontproperties='SimHei')
            self.ax_profile1.grid(alpha=0.3)
            
            self.ax_profile2.text(0.5, 0.5, '请在RGB图上画线',
                                 ha='center', va='center',
                                 transform=self.ax_profile2.transAxes,
                                 fontproperties='SimHei', fontsize=14)
            self.ax_profile2.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=11)
            self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
            self.ax_profile2.set_title('fastdepth - 剖面', fontsize=12, fontproperties='SimHei')
            self.ax_profile2.grid(alpha=0.3)
        
        self.fig.canvas.draw()
    
    def run(self):
        """启动交互式界面"""
        # 创建图形 - 使用GridSpec，2行3列，所有子图等宽
        self.fig = plt.figure(figsize=(21, 12))  # 增加宽度，让三个图更宽松
        
        # 创建GridSpec: 2行3列，所有子图等宽，设置适当的间距
        gs = self.fig.add_gridspec(2, 3, 
                                   width_ratios=[1, 1, 1],  # 三等分
                                   height_ratios=[1, 1.2],  # 下面稍高一点给剖面图
                                   hspace=0.25,            # 行间距
                                   wspace=0.2)             # 列间距
        
        # ========== 第一行：三个图均匀分布 ==========
        # RGB图
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title('RGB图 - 点击画线\n(每条线需点2个点)', 
                             fontsize=12, fontproperties='SimHei', fontweight='bold')
        self.ax_rgb.axis('off')
        
        # DepthAntThingVITS
        self.ax_depth1 = self.fig.add_subplot(gs[0, 1])
        self.ax_depth1.imshow(self.depth1, cmap='jet')
        self.ax_depth1.set_title('DepthAntThingVITS', fontsize=12, fontproperties='SimHei', fontweight='bold')
        self.ax_depth1.axis('off')
        
        # fastdepth
        self.ax_depth2 = self.fig.add_subplot(gs[0, 2])
        self.ax_depth2.imshow(self.depth2, cmap='jet')
        self.ax_depth2.set_title('fastdepth', fontsize=12, fontproperties='SimHei', fontweight='bold')
        self.ax_depth2.axis('off')
        
        # ========== 第二行：两个剖面图，各占1.5列，但为了均匀分布，我们让它们各占1.5列 ==========
        # 使用subgridspec实现更精细的控制
        gs2 = gs[1, :].subgridspec(1, 2, width_ratios=[1.5, 1.5], wspace=0.15)
        
        # DepthAntThingVITS剖面
        self.ax_profile1 = self.fig.add_subplot(gs2[0, 0])
        self.ax_profile1.text(0.5, 0.5, '请在RGB图上画线\n\n点击画线后显示深度剖面', 
                             ha='center', va='center',
                             transform=self.ax_profile1.transAxes,
                             fontproperties='SimHei', fontsize=14, 
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_profile1.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_title('DepthAntThingVITS - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile1.grid(alpha=0.3)
        
        # fastdepth剖面
        self.ax_profile2 = self.fig.add_subplot(gs2[0, 1])
        self.ax_profile2.text(0.5, 0.5, '请在RGB图上画线\n\n点击画线后显示深度剖面',
                             ha='center', va='center',
                             transform=self.ax_profile2.transAxes,
                             fontproperties='SimHei', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_profile2.set_xlabel('距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_title('fastdepth - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile2.grid(alpha=0.3)
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        plt.tight_layout()
        plt.show()


# ==================== 自由绘制模式 ====================

class FreehandDepthProfiler:
    def __init__(self, rgb_path, depth1_path, depth2_path):
        """
        自由绘制曲线查看深度剖面
        """
        self.rgb = cv2.imread(rgb_path)
        self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)
        
        self.depth1 = self._load_depth(depth1_path)
        self.depth2 = self._load_depth(depth2_path)
        
        h, w = self.rgb.shape[:2]
        self.depth1 = cv2.resize(self.depth1, (w, h))
        self.depth2 = cv2.resize(self.depth2, (w, h))
        
        # 存储用户画的路径
        self.paths = []
        self.current_path = []
        self.is_drawing = False
        
        # 图形对象
        self.fig = None
        self.ax_rgb = None
        self.ax_profile1 = None
        self.ax_profile2 = None
        
        # 保存路径
        self.save_dir = "depth_profiles_freehand"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"图像尺寸: {self.rgb.shape[:2]}")
        print("\n使用说明:")
        print("="*60)
        print("1. 按住鼠标左键拖动画线（自由曲线）")
        print("2. 松开鼠标完成一条线")
        print("3. 按 'c' 键清除所有线")
        print("4. 按 'u' 键撤销最后一条线")
        print("5. 按 's' 键手动保存")
        print("6. 按 'q' 键或关闭窗口退出并保存")
        print("="*60)
    
    def _load_depth(self, path):
        """加载并归一化深度图"""
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth.dtype == np.uint8:
            depth = depth.astype(np.float32) / 255.0
        else:
            depth = depth.astype(np.float32)
        return cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
    
    def sample_along_path(self, depth, path, num_samples=500):
        """沿着路径采样深度值"""
        if len(path) < 2:
            return None, None
        
        path = np.array(path)
        
        # 使用样条插值平滑路径
        if len(path) >= 4:
            # 参数化路径
            t = np.linspace(0, 1, len(path))
            
            # 样条插值
            try:
                fx = interpolate.interp1d(t, path[:, 0], kind='cubic')
                fy = interpolate.interp1d(t, path[:, 1], kind='cubic')
                
                # 生成平滑路径
                t_new = np.linspace(0, 1, num_samples)
                x_coords = fx(t_new)
                y_coords = fy(t_new)
            except:
                # 如果插值失败，使用线性插值
                t_new = np.linspace(0, len(path)-1, num_samples)
                x_coords = np.interp(t_new, range(len(path)), path[:, 0])
                y_coords = np.interp(t_new, range(len(path)), path[:, 1])
        else:
            # 点太少，直接线性插值
            t_new = np.linspace(0, len(path)-1, num_samples)
            x_coords = np.interp(t_new, range(len(path)), path[:, 0])
            y_coords = np.interp(t_new, range(len(path)), path[:, 1])
        
        # 采样深度值
        depth_values = []
        for x, y in zip(x_coords, y_coords):
            x = np.clip(x, 0, depth.shape[1] - 1)
            y = np.clip(y, 0, depth.shape[0] - 1)
            
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, depth.shape[1] - 1), min(y0 + 1, depth.shape[0] - 1)
            
            dx, dy = x - x0, y - y0
            
            depth_val = (1 - dx) * (1 - dy) * depth[y0, x0] + \
                       dx * (1 - dy) * depth[y0, x1] + \
                       (1 - dx) * dy * depth[y1, x0] + \
                       dx * dy * depth[y1, x1]
            
            depth_values.append(depth_val)
        
        # 计算累积距离
        distances = [0]
        for i in range(1, len(x_coords)):
            dx = x_coords[i] - x_coords[i-1]
            dy = y_coords[i] - y_coords[i-1]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(distances[-1] + dist)
        
        return np.array(depth_values), np.array(distances)
    
    def on_press(self, event):
        """鼠标按下"""
        if event.inaxes != self.ax_rgb:
            return
        
        if event.button == 1:
            self.is_drawing = True
            self.current_path = [(int(event.xdata), int(event.ydata))]
    
    def on_motion(self, event):
        """鼠标移动"""
        if not self.is_drawing or event.inaxes != self.ax_rgb:
            return
        
        x, y = int(event.xdata), int(event.ydata)
        
        # 避免重复点
        if len(self.current_path) == 0 or \
           (x, y) != self.current_path[-1]:
            self.current_path.append((x, y))
            
            # 实时绘制
            if len(self.current_path) > 1:
                p1 = self.current_path[-2]
                p2 = self.current_path[-1]
                self.ax_rgb.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                               'r-', linewidth=2, alpha=0.7)
                self.fig.canvas.draw_idle()
    
    def on_release(self, event):
        """鼠标释放"""
        if not self.is_drawing:
            return
        
        self.is_drawing = False
        
        if len(self.current_path) > 1:
            self.paths.append(self.current_path)
            self.redraw_all()
            print(f"✓ 已添加第 {len(self.paths)} 条线 ({len(self.current_path)} 个点)")
        
        self.current_path = []
    
    def on_key(self, event):
        """处理键盘事件"""
        if event.key == 'c':
            self.paths = []
            self.current_path = []
            self.redraw_all()
            print("已清除所有线")
        
        elif event.key == 'u':
            if self.paths:
                self.paths.pop()
                self.redraw_all()
                print(f"已撤销，剩余 {len(self.paths)} 条线")
        
        elif event.key == 's':
            self.save_results()
            print("手动保存完成")
        
        elif event.key == 'q':
            self.save_results()
            plt.close(self.fig)
            print("已保存并退出")
    
    def on_close(self, event):
        """窗口关闭事件"""
        self.save_results()
        print("已自动保存结果")
    
    def save_results(self):
        """保存当前结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存整个figure
        if self.fig:
            fig_path = os.path.join(self.save_dir, f"{name}.png")
            self.fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"✓ 完整视图已保存: {fig_path}")
            
            # 如果有画线，也保存剖面数据
            
            #if self.paths:
            #    data_path = os.path.join(self.save_dir, f"freehand_data_{timestamp}.txt")
            #    with open(data_path, 'w', encoding='utf-8') as f:
            #        f.write("自由绘制深度剖面数据\n")
            #        f.write("="*60 + "\n")
            #        f.write(f"保存时间: {datetime.now()}\n")
            #        f.write(f"线条数量: {len(self.paths)}\n\n")
            #        
            #        for i, path in enumerate(self.paths):
            #            depths1, distances = self.sample_along_path(self.depth1, path)
            #            depths2, _ = self.sample_along_path(self.depth2, path)
            #            
            #            f.write(f"线 {i+1}: {len(path)} 个控制点\n")
            #            f.write(f"  路径长度: {distances[-1]:.2f} 像素\n")
            #            f.write(f"  DepthAntThingVITS - 最小值: {depths1.min():.4f}, 最大值: {depths1.max():.4f}, 均值: {depths1.mean():.4f}\n")
            #            f.write(f"  fastdepth - 最小值: {depths2.min():.4f}, 最大值: {depths2.max():.4f}, 均值: {depths2.mean():.4f}\n\n")
            #    print(f"✓ 剖面数据已保存: {data_path}")
            
   
    def redraw_all(self):
        """重绘所有内容"""
        self.ax_rgb.clear()
        self.ax_profile1.clear()
        self.ax_profile2.clear()
        
        # RGB图
        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title('RGB图 - 拖动画线\n(按住左键拖动)', 
                             fontsize=12, fontproperties='SimHei', fontweight='bold')
        self.ax_rgb.axis('off')
        
        # 重新画所有路径
        for i, path in enumerate(self.paths):
            color = plt.cm.rainbow(i / max(len(self.paths), 1))
            path = np.array(path)
            self.ax_rgb.plot(path[:, 0], path[:, 1], 
                           color=color, linewidth=3, alpha=0.7)
        
        # 更新剖面图
        self.update_profiles()
    
    def update_profiles(self):
        """更新深度剖面图"""
        if not self.paths:
            self.ax_profile1.text(0.5, 0.5, '请在RGB图上画线\n\n按住左键拖动画线', 
                                 ha='center', va='center',
                                 transform=self.ax_profile1.transAxes,
                                 fontproperties='SimHei', fontsize=14,
                                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            self.ax_profile1.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
            self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
            self.ax_profile1.set_title('DepthAntThingVITS - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
            self.ax_profile1.grid(alpha=0.3)
            
            self.ax_profile2.text(0.5, 0.5, '请在RGB图上画线\n\n按住左键拖动画线',
                                 ha='center', va='center',
                                 transform=self.ax_profile2.transAxes,
                                 fontproperties='SimHei', fontsize=14,
                                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            self.ax_profile2.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
            self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
            self.ax_profile2.set_title('fastdepth - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
            self.ax_profile2.grid(alpha=0.3)
            self.fig.canvas.draw()
            return
        
        self.ax_profile1.clear()
        self.ax_profile2.clear()
        
        # 绘制每条路径的剖面
        for i, path in enumerate(self.paths):
            color = plt.cm.rainbow(i / max(len(self.paths), 1))
            
            depths1, distances = self.sample_along_path(self.depth1, path)
            depths2, _ = self.sample_along_path(self.depth2, path)
            
            if depths1 is not None:
                self.ax_profile1.plot(distances, depths1, color=color,
                                     linewidth=2.5, label=f'线 {i+1}', alpha=0.8)
                self.ax_profile2.plot(distances, depths2, color=color,
                                     linewidth=2.5, label=f'线 {i+1}', alpha=0.8)
        
        self.ax_profile1.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_title('DepthAntThingVITS - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile1.grid(alpha=0.3, linestyle='--')
        if self.paths:
            self.ax_profile1.legend(prop={'family': 'SimHei', 'size': 10}, loc='best')
        
        self.ax_profile2.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_title('fastdepth - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile2.grid(alpha=0.3, linestyle='--')
        if self.paths:
            self.ax_profile2.legend(prop={'family': 'SimHei', 'size': 10}, loc='best')
        
        # 设置相同的y轴范围
        if self.paths:
            y_min = min(self.ax_profile1.get_ylim()[0], self.ax_profile2.get_ylim()[0])
            y_max = max(self.ax_profile1.get_ylim()[1], self.ax_profile2.get_ylim()[1])
            self.ax_profile1.set_ylim(y_min, y_max)
            self.ax_profile2.set_ylim(y_min, y_max)
        
        self.fig.canvas.draw()
    
    def run(self):
        """启动交互式界面"""
        # 创建图形 - 1行3列，所有子图等宽
        self.fig = plt.figure(figsize=(21, 7))  # 21:7 = 3:1 比例
        
        # 创建GridSpec: 1行3列，完全等宽
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.2)
        
        # RGB图
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title('RGB图 - 拖动画线\n(按住左键拖动)', 
                             fontsize=12, fontproperties='SimHei', fontweight='bold')
        self.ax_rgb.axis('off')
        
        # DepthAntThingVITS剖面
        self.ax_profile1 = self.fig.add_subplot(gs[0, 1])
        self.ax_profile1.text(0.5, 0.5, '请在RGB图上画线\n\n按住左键拖动画线',
                             ha='center', va='center',
                             transform=self.ax_profile1.transAxes,
                             fontproperties='SimHei', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_profile1.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile1.set_title('DepthAntThingVITS - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile1.grid(alpha=0.3)
        
        # fastdepth剖面
        self.ax_profile2 = self.fig.add_subplot(gs[0, 2])
        self.ax_profile2.text(0.5, 0.5, '请在RGB图上画线\n\n按住左键拖动画线',
                             ha='center', va='center',
                             transform=self.ax_profile2.transAxes,
                             fontproperties='SimHei', fontsize=14,
                             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        self.ax_profile2.set_xlabel('路径距离 (像素)', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_ylabel('深度值', fontproperties='SimHei', fontsize=11)
        self.ax_profile2.set_title('fastdepth - 深度剖面', fontsize=13, fontproperties='SimHei', fontweight='bold')
        self.ax_profile2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # 连接事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        plt.show()


# ==================== 使用示例 ====================

if __name__ == "__main__":
    name=34
    rgb_image = f"src/{name}.jpg"
    depth_map1 = f"DepthAntThingVITS/{name}.png"
    depth_map2 = f"fastdepth/{name}.png"
    
    print("请选择模式:")
    print("1 - 点击模式（每条线点2个点）")
    print("2 - 自由绘制模式（拖动画线）")
    
    try:
        choice = input("输入选择 (1/2): ").strip()
        
        if choice == '2':
            # 自由绘制模式
            profiler = FreehandDepthProfiler(rgb_image, depth_map1, depth_map2)
            profiler.run()
        else:
            # 点击模式（默认）
            profiler = InteractiveDepthProfiler(rgb_image, depth_map1, depth_map2)
            profiler.run()
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()