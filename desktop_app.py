"""
出芽率检测桌面应用 — tkinter 原生 GUI，完全离线，无需浏览器
运行: python desktop_app.py
可打包为 .exe: pyinstaller --onefile --windowed desktop_app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import csv
import datetime
import io
import base64
from pathlib import Path

# ========== HSV 默认参数 ==========
DEFAULTS = {
    "g_h_min": 28, "g_h_max": 85, "g_s_min": 25, "g_s_max": 180, "g_v_min": 45, "g_v_max": 220,
    "s_h_min": 0,  "s_h_max": 32,  "s_s_min": 8,  "s_s_max": 255, "s_v_min": 10, "s_v_max": 255,
    "erode": 1, "dilate": 2,
}

DATA_FILE = str(Path(__file__).parent / "germination_record.csv")

# ========== 检测引擎 ==========
def detect(frame, params, roi=None):
    """roi: (x, y, w, h) 或 None。分母 = ROI 面积（非土壤检测）"""
    h_full, w_full = frame.shape[:2]

    if roi is not None:
        rx, ry, rw, rh = roi
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, w_full - rx), min(rh, h_full - ry)
        if rw <= 0 or rh <= 0:
            roi = None
        else:
            crop = frame[ry:ry+rh, rx:rx+rw]
            roi_area = rw * rh
    else:
        crop = frame
        roi_area = w_full * h_full

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_low  = np.array([params["g_h_min"], params["g_s_min"], params["g_v_min"]])
    green_high = np.array([params["g_h_max"], params["g_s_max"], params["g_v_max"]])
    soil_low   = np.array([params["s_h_min"], params["s_s_min"], params["s_v_min"]])
    soil_high  = np.array([params["s_h_max"], params["s_s_max"], params["s_v_max"]])

    mask_g = cv2.inRange(hsv, green_low, green_high)
    mask_s = cv2.inRange(hsv, soil_low, soil_high)

    e, d = params["erode"], params["dilate"]
    if e > 0 or d > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if e > 0: mask_g = cv2.erode(mask_g, kernel, iterations=e)
        if d > 0: mask_g = cv2.dilate(mask_g, kernel, iterations=d)

    green_area = cv2.countNonZero(mask_g)
    rate = (green_area / roi_area * 100) if roi_area > 0 else 0.0
    mask_s_only = cv2.bitwise_and(mask_s, cv2.bitwise_not(mask_g))
    soil_area = cv2.countNonZero(mask_s_only)

    overlay = crop.copy()
    overlay[mask_g > 0] = (0, 255, 0)
    overlay[mask_s_only > 0] = (0, 120, 200)
    annotated_crop = cv2.addWeighted(crop, 0.4, overlay, 0.6, 0)

    annotated = frame.copy()
    if roi is not None:
        annotated[ry:ry+rh, rx:rx+rw] = annotated_crop
        cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh), (255, 200, 0), 3)
    else:
        annotated = annotated_crop

    return rate, green_area, roi_area, mask_g, annotated


# ========== 主应用 ==========
class GerminationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🌱 出芽率智能检测")
        self.root.geometry("1100x780")
        self.root.minsize(900, 600)
        self.root.configure(bg="#0f1117")

        # 状态
        self.frame = None          # 当前 BGR 图像
        self.display_img = None    # 当前显示的 PIL Image
        self.last_result = None
        self.view_mode = "overlay"  # overlay | mask
        self.params = DEFAULTS.copy()
        # ROI 框选
        self.roi = None            # (x, y, w, h) 原图坐标
        self.roi_mode = False
        self.roi_start = None      # (x, y) 画布坐标

        self._setup_ui()
        self._load_history()

    # ========== UI 构建 ==========
    def _setup_ui(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#1a1d27")
        style.configure("TLabel", background="#1a1d27", foreground="#e0e0e0", font=("微软雅黑", 10))
        style.configure("TButton", background="#2a2d3a", foreground="#e0e0e0", font=("微软雅黑", 9))
        style.configure("TScale", background="#1a1d27")
        style.configure("Green.TLabel", foreground="#4ade80", font=("微软雅黑", 16, "bold"))
        style.configure("Big.TLabel", font=("微软雅黑", 40, "bold"), foreground="#4ade80")

        # === 主布局 ===
        self.main_frame = tk.Frame(self.root, bg="#0f1117")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧：图片区
        self.left_frame = tk.Frame(self.main_frame, bg="#1a1d27", relief=tk.FLAT, bd=0,
                                   highlightbackground="#2a2d3a", highlightthickness=1)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # 图片标题栏
        left_top = tk.Frame(self.left_frame, bg="#1a1d27")
        left_top.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(left_top, text="📷 检测结果", bg="#1a1d27", fg="#888", font=("微软雅黑", 11)).pack(side=tk.LEFT)
        btn_frame = tk.Frame(left_top, bg="#1a1d27")
        btn_frame.pack(side=tk.RIGHT)
        self.btn_overlay = tk.Button(btn_frame, text="叠加", bg="#22c55e", fg="#000",
                                     font=("微软雅黑", 9), relief=tk.FLAT, padx=10, pady=2,
                                     command=lambda: self._switch_view("overlay"))
        self.btn_overlay.pack(side=tk.LEFT, padx=2)
        self.btn_mask = tk.Button(btn_frame, text="掩膜", bg="#2a2d3a", fg="#e0e0e0",
                                  font=("微软雅黑", 9), relief=tk.FLAT, padx=10, pady=2,
                                  command=lambda: self._switch_view("mask"))
        self.btn_mask.pack(side=tk.LEFT, padx=2)

        # 图片显示区
        self.img_canvas = tk.Canvas(self.left_frame, bg="#0a0c12", highlightthickness=0)
        self.img_canvas.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        # ROI 框选事件
        self.img_canvas.bind("<ButtonPress-1>", self._roi_mousedown)
        self.img_canvas.bind("<B1-Motion>", self._roi_mousemove)
        self.img_canvas.bind("<ButtonRelease-1>", self._roi_mouseup)

        # 底部按钮
        left_bottom = tk.Frame(self.left_frame, bg="#1a1d27")
        left_bottom.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Button(left_bottom, text="📁 选择图片", bg="#22c55e", fg="#000",
                  font=("微软雅黑", 10, "bold"), relief=tk.FLAT, padx=16, pady=6,
                  command=self._load_image).pack(side=tk.LEFT, padx=(0, 8))
        self.btn_roi = tk.Button(left_bottom, text="📐 框选苗盘", bg="#2a2d3a", fg="#e0e0e0",
                                 font=("微软雅黑", 10), relief=tk.FLAT, padx=12, pady=6,
                                 command=self._toggle_roi, state=tk.DISABLED)
        self.btn_roi.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_analyze = tk.Button(left_bottom, text="🔍 分析", bg="#2a2d3a", fg="#e0e0e0",
                                     font=("微软雅黑", 10), relief=tk.FLAT, padx=16, pady=6,
                                     command=self._analyze, state=tk.DISABLED)
        self.btn_analyze.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_save = tk.Button(left_bottom, text="💾 保存记录", bg="#2a2d3a", fg="#e0e0e0",
                                  font=("微软雅黑", 10), relief=tk.FLAT, padx=16, pady=6,
                                  command=self._save_record, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT)

        # ROI 信息栏
        self.roi_info = tk.Label(left_bottom, text="", bg="#1a1d27", fg="#ffc800",
                                 font=("微软雅黑", 9))
        self.roi_info.pack(side=tk.RIGHT, padx=(10, 0))

        # === 右侧面板 ===
        self.right_frame = tk.Frame(self.main_frame, bg="#1a1d27", width=320,
                                    relief=tk.FLAT, bd=0,
                                    highlightbackground="#2a2d3a", highlightthickness=1)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.right_frame.pack_propagate(False)

        right_inner = tk.Frame(self.right_frame, bg="#1a1d27")
        right_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # 结果卡片
        result_card = tk.Frame(right_inner, bg="#1a2a1a", relief=tk.FLAT, bd=0,
                               highlightbackground="#2a2d3a", highlightthickness=1)
        result_card.pack(fill=tk.X, pady=(0, 12))

        self.rate_label = tk.Label(result_card, text="--", bg="#1a2a1a", fg="#4ade80",
                                   font=("微软雅黑", 44, "bold"))
        self.rate_label.pack(pady=(16, 0))
        tk.Label(result_card, text="出芽率", bg="#1a2a1a", fg="#888",
                 font=("微软雅黑", 10)).pack()

        stats_frame = tk.Frame(result_card, bg="#1a2a1a")
        stats_frame.pack(pady=(8, 14))
        self.green_px_label = tk.Label(stats_frame, text="绿色: - px", bg="#1a2a1a", fg="#4ade80",
                                       font=("微软雅黑", 9))
        self.green_px_label.pack(side=tk.LEFT, padx=10)
        self.soil_px_label = tk.Label(stats_frame, text="苗盘: - px", bg="#1a2a1a", fg="#d4a574",
                                      font=("微软雅黑", 9))
        self.soil_px_label.pack(side=tk.LEFT, padx=10)

        # 参数滚动区
        canvas = tk.Canvas(right_inner, bg="#1a1d27", highlightthickness=0)
        scrollbar = tk.Scrollbar(right_inner, orient=tk.VERTICAL, command=canvas.yview)
        self.param_frame = tk.Frame(canvas, bg="#1a1d27")
        self.param_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 鼠标滚轮绑定
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # 构建参数滑块
        self._build_param_section("🎨 绿色 HSV", [
            ("H 色相", "g_h_min", "g_h_max", 0, 179),
            ("S 饱和度", "g_s_min", "g_s_max", 0, 255),
            ("V 明度", "g_v_min", "g_v_max", 0, 255),
        ])
        self._build_param_section("🟤 土壤 HSV", [
            ("H 色相", "s_h_min", "s_h_max", 0, 179),
            ("S 饱和度", "s_s_min", "s_s_max", 0, 255),
            ("V 明度", "s_v_min", "s_v_max", 0, 255),
        ])
        self._build_single_slider("🔧 形态学", "Erode 腐蚀", "erode", 0, 10)
        self._build_single_slider(None, "Dilate 膨胀", "dilate", 0, 10)

        # 参数操作按钮
        param_btns = tk.Frame(self.param_frame, bg="#1a1d27")
        param_btns.pack(fill=tk.X, pady=(10, 4))
        tk.Button(param_btns, text="↺ 重置参数", bg="#2a2d3a", fg="#e0e0e0",
                  font=("微软雅黑", 9), relief=tk.FLAT, padx=10, pady=3,
                  command=self._reset_params).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(param_btns, text="🔄 应用", bg="#22c55e", fg="#000",
                  font=("微软雅黑", 9), relief=tk.FLAT, padx=10, pady=3,
                  command=self._apply_params).pack(side=tk.LEFT)

        # === 底部：历史记录 ===
        self.history_frame = tk.Frame(self.root, bg="#1a1d27",
                                      highlightbackground="#2a2d3a", highlightthickness=1)
        self.history_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        hist_header = tk.Frame(self.history_frame, bg="#1a1d27")
        hist_header.pack(fill=tk.X, padx=12, pady=(10, 4))
        tk.Label(hist_header, text="📊 检测记录", bg="#1a1d27", fg="#888",
                 font=("微软雅黑", 11)).pack(side=tk.LEFT)
        tk.Button(hist_header, text="📥 导出 CSV", bg="#2a2d3a", fg="#e0e0e0",
                  font=("微软雅黑", 9), relief=tk.FLAT, padx=10, pady=3,
                  command=self._export_csv).pack(side=tk.RIGHT)

        # 历史表格
        columns = ("time", "rate", "green", "tray")
        self.tree = ttk.Treeview(self.history_frame, columns=columns, show="headings", height=5)
        self.tree.heading("time", text="时间")
        self.tree.heading("rate", text="出芽率")
        self.tree.heading("green", text="绿色 px")
        self.tree.heading("tray", text="苗盘 px")
        self.tree.column("time", width=120, anchor="center")
        self.tree.column("rate", width=80, anchor="center")
        self.tree.column("green", width=100, anchor="center")
        self.tree.column("tray", width=100, anchor="center")
        self.tree.pack(fill=tk.X, padx=12, pady=(0, 10))

    def _build_param_section(self, title, sliders):
        """构建一组双滑块参数"""
        if title:
            sep = tk.Frame(self.param_frame, bg="#2a2d3a", height=1)
            sep.pack(fill=tk.X, pady=(10, 6))
            tk.Label(self.param_frame, text=title, bg="#1a1d27", fg="#888",
                     font=("微软雅黑", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))

        for label_text, min_key, max_key, min_v, max_v in sliders:
            row = tk.Frame(self.param_frame, bg="#1a1d27")
            row.pack(fill=tk.X, pady=2)

            tk.Label(row, text=label_text, bg="#1a1d27", fg="#999",
                     font=("微软雅黑", 9), width=8, anchor=tk.W).pack(side=tk.LEFT)

            val_label = tk.Label(row, text="", bg="#1a1d27", fg="#aaa",
                                 font=("微软雅黑", 8), width=12, anchor=tk.E)
            val_label.pack(side=tk.RIGHT, padx=(4, 0))

            def make_cmd(key, lbl):
                def cmd(val):
                    self.params[key] = int(float(val))
                    lbl.config(text=str(int(float(val))))
                return cmd

            s_min = ttk.Scale(row, from_=min_v, to=max_v, orient=tk.HORIZONTAL,
                              command=make_cmd(min_key, val_label))
            s_min.set(self.params[min_key])
            s_min.pack(side=tk.LEFT, fill=tk.X, expand=True)

            s_max = ttk.Scale(row, from_=min_v, to=max_v, orient=tk.HORIZONTAL,
                              command=make_cmd(max_key, val_label))
            s_max.set(self.params[max_key])
            s_max.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # 存储引用
            setattr(self, f"slider_{min_key}", s_min)
            setattr(self, f"slider_{max_key}", s_max)
            setattr(self, f"label_{min_key}_{max_key}", val_label)
            val_label.config(text=f"{self.params[min_key]} - {self.params[max_key]}")

    def _build_single_slider(self, title, label_text, key, min_v, max_v):
        if title:
            sep = tk.Frame(self.param_frame, bg="#2a2d3a", height=1)
            sep.pack(fill=tk.X, pady=(10, 6))
            tk.Label(self.param_frame, text=title, bg="#1a1d27", fg="#888",
                     font=("微软雅黑", 9, "bold")).pack(anchor=tk.W, pady=(0, 4))

        row = tk.Frame(self.param_frame, bg="#1a1d27")
        row.pack(fill=tk.X, pady=2)

        tk.Label(row, text=label_text, bg="#1a1d27", fg="#999",
                 font=("微软雅黑", 9), width=10, anchor=tk.W).pack(side=tk.LEFT)

        val_label = tk.Label(row, text="", bg="#1a1d27", fg="#aaa",
                             font=("微软雅黑", 8), width=4, anchor=tk.E)
        val_label.pack(side=tk.RIGHT)

        def cmd(val):
            self.params[key] = int(float(val))
            val_label.config(text=str(int(float(val))))

        s = ttk.Scale(row, from_=min_v, to=max_v, orient=tk.HORIZONTAL, command=cmd)
        s.set(self.params[key])
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        setattr(self, f"slider_{key}", s)
        val_label.config(text=str(self.params[key]))

    # ========== 功能方法 ==========
    def _switch_view(self, view):
        self.view_mode = view
        if view == "overlay":
            self.btn_overlay.config(bg="#22c55e", fg="#000")
            self.btn_mask.config(bg="#2a2d3a", fg="#e0e0e0")
        else:
            self.btn_overlay.config(bg="#2a2d3a", fg="#e0e0e0")
            self.btn_mask.config(bg="#22c55e", fg="#000")
        if self.last_result:
            self._show_result(self.last_result)

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="选择育苗盘照片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("错误", "无法读取图片")
            return
        self.frame = img
        self.last_result = None
        self.roi = None
        self.roi_mode = False
        self.btn_roi.config(text="📐 框选苗盘", bg="#2a2d3a", fg="#e0e0e0")
        self.roi_info.config(text="")
        self._display_frame(img)
        self.btn_analyze.config(state=tk.NORMAL)
        self.btn_roi.config(state=tk.NORMAL)
        self.rate_label.config(text="--")
        self.green_px_label.config(text="绿色: - px")
        self.soil_px_label.config(text="苗盘: - px")
        self.btn_save.config(state=tk.DISABLED)

    def _analyze(self):
        if self.frame is None:
            return
        self.btn_analyze.config(text="⏳ 分析中...", state=tk.DISABLED)
        self.root.update()

        rate, ga, roi_a, mask_g, annotated = detect(self.frame, self.params, roi=self.roi)
        self.last_result = {
            "rate": round(rate, 1),
            "green_area": ga,
            "roi_area": roi_a,
            "mask_g": mask_g,
            "annotated": annotated,
        }
        self._show_result(self.last_result)
        self.btn_analyze.config(text="🔍 分析", state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)

    def _show_result(self, result):
        self.rate_label.config(text=f"{result['rate']}%")
        self.green_px_label.config(text=f"绿色: {result['green_area']:,} px")
        self.soil_px_label.config(text=f"苗盘: {result['roi_area']:,} px")

        if self.view_mode == "overlay":
            self._display_frame(result["annotated"])
        else:
            mask_bgr = cv2.cvtColor(result["mask_g"], cv2.COLOR_GRAY2BGR)
            self._display_frame(mask_bgr)

    def _display_frame(self, bgr_img):
        """在 Canvas 中显示 BGR 图像（等比缩放适配），并绘制 ROI 框"""
        if bgr_img is None:
            return
        h, w = bgr_img.shape[:2]
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        if cw < 50: cw = 600
        if ch < 50: ch = 400

        scale = min((cw - 10) / w, (ch - 10) / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr_img, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.display_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.img_canvas.delete("all")
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self.img_canvas.create_image(cw // 2, ch // 2, image=self.display_img, anchor=tk.CENTER)

        # 绘制 ROI 框
        if self.roi:
            rx, ry, rw, rh = self.roi
            cx0 = ox + int(rx * scale)
            cy0 = oy + int(ry * scale)
            cx1 = ox + int((rx + rw) * scale)
            cy1 = oy + int((ry + rh) * scale)
            # 半透明填充
            self.img_canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                             outline="#ffc800", width=2, dash=(8, 4))
            self.img_canvas.create_rectangle(cx0, cy0, cx1, cy1,
                                             fill="#ffff00", stipple="gray12", outline="")

    def _save_record(self):
        if not self.last_result:
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = not os.path.exists(DATA_FILE)
        with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(["时间", "出芽率%", "绿色像素", "苗盘像素"])
            w.writerow([now, str(self.last_result["rate"]),
                        self.last_result["green_area"], self.last_result["roi_area"]])
        self._load_history()
        self.btn_save.config(text="✅ 已保存")
        self.root.after(1500, lambda: self.btn_save.config(text="💾 保存记录"))

    def _load_history(self):
        self.tree.delete(*self.tree.get_children())
        if not os.path.exists(DATA_FILE):
            return
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in list(reader)[-30:]:
                self.tree.insert("", 0,
                    values=(row.get("时间", "")[-8:], row.get("出芽率%", "") + "%",
                            row.get("绿色像素", ""), row.get("苗盘像素", row.get("土壤像素", ""))))

    def _export_csv(self):
        if not os.path.exists(DATA_FILE):
            messagebox.showinfo("提示", "暂无数据")
            return
        dest = filedialog.asksaveasfilename(
            title="导出 CSV",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")]
        )
        if dest:
            import shutil
            shutil.copy(DATA_FILE, dest)
            messagebox.showinfo("成功", f"已导出到:\n{dest}")

    def _reset_params(self):
        self.params = DEFAULTS.copy()
        for key, val in DEFAULTS.items():
            slider = getattr(self, f"slider_{key}", None)
            if slider:
                slider.set(val)
            # 更新双滑块标签
            if key.endswith("_min"):
                base = key[:-4]
                max_key = base + "_max"
                if max_key in DEFAULTS:
                    lbl = getattr(self, f"label_{key}_{max_key}", None)
                    if lbl:
                        lbl.config(text=f"{val} - {DEFAULTS[max_key]}")
            elif key.endswith("_max"):
                base = key[:-4]
                min_key = base + "_min"
                if min_key in DEFAULTS:
                    lbl = getattr(self, f"label_{min_key}_{key}", None)
                    if lbl:
                        lbl.config(text=f"{DEFAULTS[min_key]} - {val}")
        if self.frame is not None:
            self._analyze()

    def _apply_params(self):
        if self.frame is not None:
            self._analyze()

    # ========== ROI 框选 ==========
    def _toggle_roi(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.btn_roi.config(text="📐 框选中...", bg="#ffc800", fg="#000")
        else:
            self.btn_roi.config(text="📐 框选苗盘", bg="#2a2d3a", fg="#e0e0e0")
            self.roi = None
            self.roi_info.config(text="")

    def _roi_mousedown(self, event):
        if not self.roi_mode:
            return
        self.roi_start = (event.x, event.y)

    def _roi_mousemove(self, event):
        if not self.roi_mode or self.roi_start is None:
            return
        self._display_frame(self.last_result["annotated"] if self.last_result else self.frame)
        x0, y0 = self.roi_start
        self.img_canvas.create_rectangle(x0, y0, event.x, event.y,
                                         outline="#ffc800", width=2, dash=(6, 3))

    def _roi_mouseup(self, event):
        if not self.roi_mode or self.roi_start is None:
            return
        x0, y0 = self.roi_start
        x1, y1 = event.x, event.y
        self.roi_start = None
        w_box, h_box = abs(x1 - x0), abs(y1 - y0)
        if w_box < 10 or h_box < 10:
            return
        # 转换为原图坐标
        if self.frame is None:
            return
        cw = self.img_canvas.winfo_width()
        ch = self.img_canvas.winfo_height()
        h_img, w_img = self.frame.shape[:2]
        scale = min((cw - 10) / w_img, (ch - 10) / h_img, 1.0)
        nw, nh = int(w_img * scale), int(h_img * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        rx = int((min(x0, x1) - ox) / scale)
        ry = int((min(y0, y1) - oy) / scale)
        rw = int(w_box / scale)
        rh = int(h_box / scale)
        self.roi = (rx, ry, rw, rh)
        self.roi_info.config(text=f"📐 苗盘: {rw}×{rh} px ({rw*rh:,} px²)")
        # 重绘以显示 ROI
        self._display_frame(self.last_result["annotated"] if self.last_result else self.frame)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = GerminationApp()
    app.run()
