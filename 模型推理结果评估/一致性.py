"""
双模型深度平面一致性直观对比工具（离散点图版）
================================================

功能清单：
- 左侧大RGB，右侧上下排列深度A/B，仅在RGB区域标注
- 鼠标悬停深度图显示数值
- 数字键1~9切换平面，m切换点/框选模式
- 按 u 撤销上一步添加的点或矩形（带屏幕提示）
- 按 q 退出，自动保存结果图（标注画布+数据摘要+散点图）
- 所有配置参数集中在文件开头
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import io
from PIL import Image

# ======================== 用户配置区 ========================
IMAGE_PATH = "test.jpg"
DEPTH_A_PATH = "depth_A.png"
DEPTH_B_PATH = "depth_B.png"

DEPTH_SCALE_A = 0.001          # PNG深度缩放因子（单位转换到米）
DEPTH_SCALE_B = 0.001

# 可视化设置
JITTER = 0.1                  # 点抖动幅度，避免重叠
SHOW_MEAN_STD = True          # 显示均值±标准差区间

# ---------- 显示尺寸控制 ----------
RGB_DISPLAY_WIDTH = 1600      # 左侧RGB显示宽度（像素），越大越清晰
CANVAS_MAX_WIDTH = 2200       # 整个画布最大宽度（若超限自动等比缩小）

# ---------- 结果保存设置 ----------
SAVE_RESULT_PATH = "depth_analysis_result.png"  # 保存文件名
SAVE_RESULT = True            # 是否保存最终结果图
# ============================================================

def load_depth(path, scale=1.0):
    """加载深度图（支持.npy和.png）"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        depth = np.load(path).astype(np.float32)
    elif ext in ['.png', '.jpg', '.jpeg']:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"无法读取: {path}")
        depth = depth.astype(np.float32) * scale
    else:
        raise ValueError(f"不支持格式: {ext}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth

def normalize_depth_for_display(depth):
    """深度图归一化到0-255用于显示（线性拉伸）"""
    valid = depth[depth > 0]
    if len(valid) == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    d_min, d_max = valid.min(), valid.max()
    if d_max - d_min < 1e-6:
        display = np.full(depth.shape, 128, dtype=np.uint8)
    else:
        display = np.clip((depth - d_min) / (d_max - d_min) * 255, 0, 255).astype(np.uint8)
    return display

class Annotator:
    """交互标注器：左侧大RGB，右侧上下排列深度A、深度B（高度严格对齐）"""
    def __init__(self, rgb_img, depth_a, depth_b):
        self.rgb_orig = rgb_img.copy()
        self.depth_a_orig = depth_a
        self.depth_b_orig = depth_b
        self.orig_h, self.orig_w = rgb_img.shape[:2]
        
        # ---------- 1. 左侧RGB缩放（固定宽度，高度等比例）----------
        self.rgb_display_w = RGB_DISPLAY_WIDTH
        self.rgb_display_h = int(self.orig_h * self.rgb_display_w / self.orig_w)
        rgb_display = cv2.resize(self.rgb_orig, (self.rgb_display_w, self.rgb_display_h))
        
        # ---------- 2. 右侧深度图：上下排列，总高度 = 左侧高度 ----------
        target_right_height = self.rgb_display_h
        depth_a_h = target_right_height // 2
        depth_b_h = target_right_height - depth_a_h
        
        depth_a_w = int(self.orig_w * depth_a_h / self.orig_h)
        depth_b_w = int(self.orig_w * depth_b_h / self.orig_h)
        right_width = max(depth_a_w, depth_b_w)  # 统一宽度
        
        # 归一化深度图并转为彩色
        depth_a_disp = normalize_depth_for_display(depth_a)
        depth_b_disp = normalize_depth_for_display(depth_b)
        depth_a_color = cv2.cvtColor(depth_a_disp, cv2.COLOR_GRAY2BGR)
        depth_b_color = cv2.cvtColor(depth_b_disp, cv2.COLOR_GRAY2BGR)
        
        # 缩放深度图至目标尺寸
        depth_a_display = cv2.resize(depth_a_color, (right_width, depth_a_h))
        depth_b_display = cv2.resize(depth_b_color, (right_width, depth_b_h))
        
        # 垂直堆叠两个深度图
        right_panel = np.vstack([depth_a_display, depth_b_display])
        
        # ---------- 3. 水平拼接左侧RGB和右侧深度面板 ----------
        canvas = np.hstack([rgb_display, right_panel])
        self.canvas_h, self.canvas_w = canvas.shape[:2]
        
        # ---------- 4. 如果总宽度超限，整体等比缩小 ----------
        if self.canvas_w > CANVAS_MAX_WIDTH:
            scale = CANVAS_MAX_WIDTH / self.canvas_w
            new_w = int(self.canvas_w * scale)
            new_h = int(self.canvas_h * scale)
            canvas = cv2.resize(canvas, (new_w, new_h))
            self.canvas_h, self.canvas_w = canvas.shape[:2]
            # 更新内部显示尺寸
            self.rgb_display_w = int(self.rgb_display_w * scale)
            self.rgb_display_h = int(self.rgb_display_h * scale)
            right_width = int(right_width * scale)
            depth_a_h = int(depth_a_h * scale)
            depth_b_h = int(depth_b_h * scale)
        
        self.canvas = canvas
        self.right_width = right_width
        self.depth_a_h = depth_a_h
        self.depth_b_h = depth_b_h
        
        # 记录各区域范围
        self.rgb_x_end = self.rgb_display_w
        self.depth_top_y_start = 0
        self.depth_top_y_end = depth_a_h
        self.depth_bottom_y_start = depth_a_h
        self.depth_bottom_y_end = depth_a_h + depth_b_h
        
        # ---------- 标注数据 ----------
        self.planes = {}
        self.current = 1
        self.mode = 'rect'
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.planes[1] = {'points': [], 'rects': [], 'color': (0,255,0)}
        
        # ---------- 撤销功能 ----------
        self.last_operation = None  # (type, pid)
        
        # 保存最终画布
        self.final_canvas = None
        
        # 窗口初始化
        self.win = "RGB (large) + DepthA (top-right) + DepthB (bottom-right)"
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.mouse_callback)

    def _canvas_to_orig(self, x_cvs, y_cvs):
        """将画布坐标（仅在RGB区域内）转换为原始RGB图像坐标"""
        if x_cvs < 0 or x_cvs >= self.rgb_x_end:
            return None, None
        if y_cvs < 0 or y_cvs >= self.rgb_display_h:
            return None, None
        x_orig = int(x_cvs * self.orig_w / self.rgb_display_w)
        y_orig = int(y_cvs * self.orig_h / self.rgb_display_h)
        x_orig = max(0, min(x_orig, self.orig_w-1))
        y_orig = max(0, min(y_orig, self.orig_h-1))
        return x_orig, y_orig

    def _orig_to_canvas(self, x_orig, y_orig):
        """将原始RGB坐标转换为画布坐标（仅RGB区域）"""
        x_cvs = int(x_orig * self.rgb_display_w / self.orig_w)
        y_cvs = int(y_orig * self.rgb_display_h / self.orig_h)
        return x_cvs, y_cvs

    def _draw_annotations(self, base_img):
        """在base_img上绘制所有标注和状态栏"""
        img_show = base_img.copy()
        
        # 绘制已有标注
        for pid, pdata in self.planes.items():
            col = pdata['color']
            for (x_orig, y_orig) in pdata['points']:
                x_d, y_d = self._orig_to_canvas(x_orig, y_orig)
                cv2.circle(img_show, (x_d, y_d), 3, col, -1)
            for (x1o, y1o, x2o, y2o) in pdata['rects']:
                x1d, y1d = self._orig_to_canvas(x1o, y1o)
                x2d, y2d = self._orig_to_canvas(x2o, y2o)
                cv2.rectangle(img_show, (x1d, y1d), (x2d, y2d), col, 2)
        
        # 状态栏
        cv2.putText(img_show, f'Plane {self.current}', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.planes[self.current]['color'], 2)
        cv2.putText(img_show, f'Mode: {self.mode}', (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        
        return img_show

    def _update_display(self, event=None, x_cvs=None, y_cvs=None, message=None):
        """更新显示：绘制标注、深度悬停、临时消息"""
        canvas_show = self._draw_annotations(self.canvas)
        
        # 显示临时消息（如撤销反馈）
        if message:
            cv2.putText(canvas_show, message, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
        # 深度图悬停数值显示
        if event == cv2.EVENT_MOUSEMOVE and x_cvs is not None and y_cvs is not None:
            if x_cvs >= self.rgb_x_end and self.depth_top_y_start <= y_cvs < self.depth_top_y_end:
                x_rel = int((x_cvs - self.rgb_x_end) * self.orig_w / self.right_width)
                y_rel = int(y_cvs * self.orig_h / self.depth_a_h)
                if 0 <= x_rel < self.orig_w and 0 <= y_rel < self.orig_h:
                    val = self.depth_a_orig[y_rel, x_rel]
                    if val > 0 and np.isfinite(val):
                        text = f'DepthA: {val:.3f}'
                        cv2.putText(canvas_show, text, (x_cvs+10, y_cvs-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            elif x_cvs >= self.rgb_x_end and self.depth_bottom_y_start <= y_cvs < self.depth_bottom_y_end:
                x_rel = int((x_cvs - self.rgb_x_end) * self.orig_w / self.right_width)
                y_rel = int((y_cvs - self.depth_bottom_y_start) * self.orig_h / self.depth_b_h)
                if 0 <= x_rel < self.orig_w and 0 <= y_rel < self.orig_h:
                    val = self.depth_b_orig[y_rel, x_rel]
                    if val > 0 and np.isfinite(val):
                        text = f'DepthB: {val:.3f}'
                        cv2.putText(canvas_show, text, (x_cvs+10, y_cvs-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        cv2.imshow(self.win, canvas_show)
        self.final_canvas = canvas_show

    def mouse_callback(self, event, x_cvs, y_cvs, flags, param):
        # 更新显示（包含标注和悬停）
        self._update_display(event, x_cvs, y_cvs)
        
        # 鼠标交互：仅在RGB区域内进行标注
        if x_cvs < 0 or x_cvs >= self.rgb_x_end or y_cvs < 0 or y_cvs >= self.rgb_display_h:
            return
        
        x_orig, y_orig = self._canvas_to_orig(x_cvs, y_cvs)
        if x_orig is None:
            return
        
        if self.mode == 'point':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.planes[self.current]['points'].append((x_orig, y_orig))
                self.last_operation = ('point', self.current)
                print(f"平面{self.current} 点 ({x_orig},{y_orig})")
        else:  # rect mode
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.ix, self.iy = x_cvs, y_cvs
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                x1o, y1o = self._canvas_to_orig(self.ix, self.iy)
                x2o, y2o = self._canvas_to_orig(x_cvs, y_cvs)
                if x1o is not None and x2o is not None:
                    self.planes[self.current]['rects'].append((x1o, y1o, x2o, y2o))
                    self.last_operation = ('rect', self.current)
                    print(f"平面{self.current} 矩形 ({x1o},{y1o})->({x2o},{y2o})")

    def undo(self):
        """撤销上一步操作（带屏幕提示）"""
        if self.last_operation is None:
            print("没有可撤销的操作")
            self._update_display(message="无操作可撤销")
            return
        op_type, pid = self.last_operation
        if pid not in self.planes:
            self.last_operation = None
            return
        
        removed_info = ""
        if op_type == 'point':
            if self.planes[pid]['points']:
                removed = self.planes[pid]['points'].pop()
                removed_info = f"点 {removed}"
                print(f"撤销平面{pid} {removed_info}")
        elif op_type == 'rect':
            if self.planes[pid]['rects']:
                removed = self.planes[pid]['rects'].pop()
                removed_info = f"矩形 {removed}"
                print(f"撤销平面{pid} {removed_info}")
        
        self.last_operation = None  # 仅支持一步撤销
        self.drawing = False       # 强制结束拖拽状态
        self._update_display(message=f"撤销成功: {removed_info}")

    def run(self):
        print("\n操作指南：")
        print("  - 左侧大尺寸RGB（标注区域），右侧上下为深度A、深度B（参考）")
        print("  - 鼠标左键拖拽：在 RGB 图上框选同一平面上的物体")
        print("  - 鼠标悬停在深度图上时显示该点深度值")
        print("  - 数字键 1~9：切换/创建新平面")
        print("  - 键 m：切换点选模式（不推荐，框选更准）")
        print("  - 键 u：撤销上一步添加的点或矩形（画布会显示提示）")
        print("  - 键 q：退出并分析\n")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.mode = 'point' if self.mode == 'rect' else 'rect'
                print(f"模式切换为 {self.mode}")
            elif key == ord('u'):
                self.undo()
            elif ord('1') <= key <= ord('9'):
                pid = key - ord('0')
                if pid not in self.planes:
                    color = tuple(np.random.randint(50,255,3).tolist())
                    self.planes[pid] = {'points':[], 'rects':[], 'color':color}
                self.current = pid
                print(f"切换到平面 {pid}")
        cv2.destroyAllWindows()
        return self.planes

    def get_final_canvas(self):
        """获取最后一次绘制的画布（包含所有标注）"""
        if self.final_canvas is None:
            self._update_display()
        return self.final_canvas.copy()

def extract_depth_samples(depth_map, planes, valid_thresh=1e-6):
    """提取每个平面所有标注区域的深度值（矩形取中位数）"""
    samples = {}
    for pid, pdata in planes.items():
        vals = []
        for x,y in pdata['points']:
            if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
                d = depth_map[y, x]
                if d > valid_thresh and np.isfinite(d):
                    vals.append(d)
        for x1,y1,x2,y2 in pdata['rects']:
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            x1 = max(0, x1); x2 = min(depth_map.shape[1], x2)
            y1 = max(0, y1); y2 = min(depth_map.shape[0], y2)
            if x2 > x1 and y2 > y1:
                region = depth_map[y1:y2, x1:x2]
                valid = region[(region > valid_thresh) & np.isfinite(region)]
                if len(valid) > 0:
                    vals.append(np.median(valid))
        samples[pid] = np.array(vals)
    return samples

def plot_strip_comparison(samples_A, samples_B, planes):
    """核心可视化：每个平面一个子图，左右并列散点图（带抖动）"""
    pids = sorted([pid for pid in samples_A if len(samples_A[pid])>0 and len(samples_B[pid])>0])
    if not pids:
        print("没有足够的有效标注数据。")
        return None

    n = len(pids)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]

    for ax, pid in zip(axes, pids):
        depA = samples_A[pid]
        depB = samples_B[pid]
        color = np.array(planes[pid]['color'])/255.0

        xA = np.random.normal(0, JITTER, size=len(depA))
        xB = np.random.normal(1, JITTER, size=len(depB))

        ax.scatter(xA, depA, alpha=0.7, s=60, c=[color], edgecolors='black', linewidth=0.5, label='Model A')
        ax.scatter(xB, depB, alpha=0.7, s=60, c=[color], edgecolors='black', linewidth=0.5, marker='s', label='Model B')

        if SHOW_MEAN_STD:
            meanA, stdA = np.mean(depA), np.std(depA)
            meanB, stdB = np.mean(depB), np.std(depB)
            ax.hlines(meanA, -0.3, 0.3, colors='blue', linestyles='-', linewidth=2)
            ax.hlines(meanB, 0.7, 1.3, colors='red', linestyles='-', linewidth=2)
            ax.fill_between([-0.3, 0.3], meanA-stdA, meanA+stdA, color='blue', alpha=0.2)
            ax.fill_between([0.7, 1.3], meanB-stdB, meanB+stdB, color='red', alpha=0.2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Model A', 'Model B'])
        ax.set_ylabel('Depth Value')
        ax.set_title(f'Plane {pid}  (n={len(depA)} objects)')
        ax.set_xlim(-0.5, 1.5)
        ax.grid(axis='y', linestyle=':', alpha=0.6)

        stdA, stdB = np.std(depA), np.std(depB)
        if stdA < stdB:
            ax.text(0.5, 0.95, '✓ Model A more consistent', transform=ax.transAxes,
                    fontsize=11, color='green', ha='center', va='top', weight='bold')
        elif stdB < stdA:
            ax.text(0.5, 0.95, '✓ Model B more consistent', transform=ax.transAxes,
                    fontsize=11, color='green', ha='center', va='top', weight='bold')
        else:
            ax.text(0.5, 0.95, 'Equal consistency', transform=ax.transAxes,
                    fontsize=11, color='gray', ha='center', va='top', weight='bold')

    plt.tight_layout()
    plt.suptitle('Depth Consistency Comparison (Strip Plot)', fontsize=14, y=1.05)
    return fig

def fig_to_cv2(fig):
    """将matplotlib figure转换为OpenCV图像（BGR格式）"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_pil = Image.open(buf)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    buf.close()
    plt.close(fig)
    return img_cv

def create_summary_image(summary_lines):
    """生成摘要表格图像（白底黑字）"""
    font_scale = 0.7
    thickness = 1
    line_height = 25
    margin = 10
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    
    max_width = 0
    for line in summary_lines:
        (w, h), _ = cv2.getTextSize(line, font_face, font_scale, thickness)
        max_width = max(max_width, w)
    
    img_w = max_width + 2 * margin
    img_h = len(summary_lines) * line_height + 2 * margin
    img = np.full((img_h, img_w, 3), 255, dtype=np.uint8)
    
    for i, line in enumerate(summary_lines):
        y = margin + i * line_height + 20
        cv2.putText(img, line, (margin, y), font_face, font_scale, (0,0,0), thickness, cv2.LINE_AA)
    
    return img

def save_composite_result(canvas_img, summary_img, scatter_img, save_path):
    """垂直拼接三部分并保存"""
    if canvas_img is None or summary_img is None or scatter_img is None:
        print("警告：部分图像为空，无法保存。")
        return
    
    target_width = canvas_img.shape[1]
    
    summary_h, summary_w = summary_img.shape[:2]
    if summary_w != target_width:
        new_h = int(summary_h * target_width / summary_w)
        summary_img = cv2.resize(summary_img, (target_width, new_h))
    
    scatter_h, scatter_w = scatter_img.shape[:2]
    if scatter_w != target_width:
        new_h = int(scatter_h * target_width / scatter_w)
        scatter_img = cv2.resize(scatter_img, (target_width, new_h))
    
    composite = np.vstack([canvas_img, summary_img, scatter_img])
    cv2.imwrite(save_path, composite)
    print(f"完整结果图已保存至: {save_path}")

def main():
    name=34
    # 请根据实际路径修改下方三行
    IMAGE_PATH = f'src/{name}.jpg'
    DEPTH_A_PATH = f'DepthAntThingVITS/{name}.png'
    DEPTH_B_PATH = f'fastdepth/{name}.png'

    # 加载数据
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"无法加载图像: {IMAGE_PATH}")
        return
    try:
        depthA = load_depth(DEPTH_A_PATH, DEPTH_SCALE_A)
        depthB = load_depth(DEPTH_B_PATH, DEPTH_SCALE_B)
    except Exception as e:
        print(f"深度图加载失败: {e}")
        return

    # 确保尺寸与RGB一致
    h, w = img.shape[:2]
    if depthA.shape[:2] != (h, w):
        depthA = cv2.resize(depthA, (w, h))
    if depthB.shape[:2] != (h, w):
        depthB = cv2.resize(depthB, (w, h))

    # ========== 启动标注窗口（左侧RGB，右侧上下深度图）==========
    annotator = Annotator(img, depthA, depthB)
    planes = annotator.run()
    canvas_img = annotator.get_final_canvas()

    # ========== 提取深度样本 ==========
    samplesA = extract_depth_samples(depthA, planes)
    samplesB = extract_depth_samples(depthB, planes)

    # ========== 绘图对比 ==========
    fig = plot_strip_comparison(samplesA, samplesB, planes)
    
    # ========== 控制台摘要 ==========
    print("\n【各平面深度一致性摘要】")
    header = "PLANE tSAMPLES DATVITSSTD tFastdepthSTD betterModel"
    print(header)
    summary_lines = [header]
    for pid in sorted(planes.keys()):
        if pid in samplesA and pid in samplesB and len(samplesA[pid])>0 and len(samplesB[pid])>0:
            stdA = np.std(samplesA[pid])
            stdB = np.std(samplesB[pid])
            better = 'A' if stdA < stdB else 'B' if stdB < stdA else '='
            line = f"{pid}           {len(samplesA[pid])}      {stdA:.4f}       {stdB:.4f}      {better}"
            print(line)
            summary_lines.append(line)
    
    # ========== 保存完整结果图 ==========
    if SAVE_RESULT and fig is not None:
        scatter_img = fig_to_cv2(fig)
        summary_img = create_summary_image(summary_lines)
        save_composite_result(canvas_img, summary_img, scatter_img, f"{name}_result.png")
    elif SAVE_RESULT:
        print("散点图为空，无法保存完整结果图。")

if __name__ == '__main__':
    main()
        
          