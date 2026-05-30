"""
HSV 参数标定工具 — 左图右参，只调 Hue 关键参数
用法: python calibrator.py [图片路径]
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
import sys
import os

# ——— 默认参数 ———
G_H_LOW  = 28
G_H_HIGH = 85
S_H_LOW  = 0
S_H_HIGH = 32
ERODE    = 1
DILATE   = 2

# 固定不变的部分（与 detector.py 同步）
G_SAT = (25, 180)
G_VAL = (45, 220)
S_SAT = (8, 255)
S_VAL = (10, 255)

# ——— 窗口配置 ———
WIN_W  = 1100
WIN_H  = 660    # 图像 ~540 + 滑块 ~120
PANEL_W = 300   # 右侧参数面板宽度


def nothing(x):
    pass


def pick_image():
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="选择一张育苗盘照片",
        filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


def build_panel(h_panel, g_h_min, g_h_max, s_h_min, s_h_max,
                erode, dilate, rate, green_area, soil_area, view_mode):
    """构建右侧参数面板"""
    panel = np.zeros((h_panel, PANEL_W, 3), dtype=np.uint8)
    panel[:, :] = (30, 30, 35)  # 深灰底色

    y = 15

    # 标题
    cv2.putText(panel, "PARAMETERS", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    y += 40

    # 出芽率大字
    cv2.putText(panel, f"{rate:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(panel, "Germination Rate", (10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 60

    # 像素统计
    cv2.putText(panel, f"Green: {green_area} px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(panel, f"Soil:  {soil_area} px", (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 0), 1)
    y += 50

    # 分割线
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    # 绿色 Hue 色带
    panel, y = draw_hue_bar_into(panel, g_h_min, g_h_max, "GREEN H", y)
    y += 5

    # 土壤 Hue 色带
    panel, y = draw_hue_bar_into(panel, s_h_min, s_h_max, "SOIL H", y)
    y += 10

    # 分割线
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    # 形态学参数
    cv2.putText(panel, f"Erode:  {erode}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(panel, f"Dilate: {dilate}", (10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    y += 55

    # 分割线
    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    # 视图提示
    if view_mode == 1:
        cv2.putText(panel, "View: Overlay", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv2.putText(panel, "View: Green Mask", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

    cv2.putText(panel, "Press 1/2 to switch", (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # 底部快捷键
    cv2.putText(panel, "s=print  q=quit", (10, h_panel - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return panel


def draw_hue_bar_into(panel, low, high, label, y0):
    """在 panel 上直接画 Hue 色带"""
    bar_y = y0 + 20
    bar_h = 24

    # 色带
    for x in range(PANEL_W - 20):
        hue = int(x / (PANEL_W - 20) * 179)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cv2.line(panel, (10 + x, bar_y), (10 + x, bar_y + bar_h),
                 (int(color[0]), int(color[1]), int(color[2])), 1)

    # 范围框
    left_x = 10 + int(low / 179 * (PANEL_W - 20))
    right_x = 10 + int(high / 179 * (PANEL_W - 20))
    cv2.rectangle(panel, (left_x, bar_y - 3), (right_x, bar_y + bar_h + 3),
                  (255, 255, 255), 2)

    # 标签
    cv2.putText(panel, label, (10, bar_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"H: {low}-{high}", (10, bar_y + bar_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return panel, bar_y + bar_h + 30


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = pick_image()

    if not path or not os.path.exists(path):
        print("❌ 未选择图片")
        return

    img = cv2.imread(path)
    if img is None:
        print(f"❌ 无法读取: {path}")
        return

    h, w = img.shape[:2]
    print(f"✅ 加载: {os.path.basename(path)}  ({w}x{h})")

    cv2.namedWindow("Calibrate", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibrate", WIN_W, WIN_H)

    cv2.createTrackbar("G_H_min", "Calibrate", G_H_LOW,  179, nothing)
    cv2.createTrackbar("G_H_max", "Calibrate", G_H_HIGH, 179, nothing)
    cv2.createTrackbar("S_H_min", "Calibrate", S_H_LOW,  179, nothing)
    cv2.createTrackbar("S_H_max", "Calibrate", S_H_HIGH, 179, nothing)
    cv2.createTrackbar("Erode",   "Calibrate", ERODE,     10, nothing)
    cv2.createTrackbar("Dilate",  "Calibrate", DILATE,    10, nothing)

    print("\n=== 操作说明 ===")
    print("  1/2 切换视图（叠加 / 绿色掩膜）")
    print("  s   打印参数")
    print("  q   退出")

    view_mode = 1

    # 计算图像可用高度（减去滑块区域）
    img_area_w = WIN_W - PANEL_W  # 800
    img_area_h = WIN_H - 130      # ~530

    while True:
        g_h_min = cv2.getTrackbarPos("G_H_min", "Calibrate")
        g_h_max = cv2.getTrackbarPos("G_H_max", "Calibrate")
        s_h_min = cv2.getTrackbarPos("S_H_min", "Calibrate")
        s_h_max = cv2.getTrackbarPos("S_H_max", "Calibrate")
        erode   = max(cv2.getTrackbarPos("Erode",  "Calibrate"), 0)
        dilate  = max(cv2.getTrackbarPos("Dilate", "Calibrate"), 0)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask_g = cv2.inRange(hsv,
            np.array([g_h_min, G_SAT[0], G_VAL[0]]),
            np.array([g_h_max, G_SAT[1], G_VAL[1]]))
        mask_s = cv2.inRange(hsv,
            np.array([s_h_min, S_SAT[0], S_VAL[0]]),
            np.array([s_h_max, S_SAT[1], S_VAL[1]]))

        if erode > 0 or dilate > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_g = cv2.erode(mask_g, kernel, iterations=erode)
            mask_g = cv2.dilate(mask_g, kernel, iterations=dilate)

        green_area = cv2.countNonZero(mask_g)
        # 从土壤 mask 中排除已被绿色覆盖的像素
        mask_s_only = cv2.bitwise_and(mask_s, cv2.bitwise_not(mask_g))
        soil_area = cv2.countNonZero(mask_s_only)
        total_area = green_area + soil_area
        rate = (green_area / total_area * 100) if total_area > 0 else 0.0

        # 左侧：图像视图
        overlay = img.copy()
        overlay[mask_g > 0] = (0, 255, 0)
        overlay[mask_s_only > 0] = (0, 120, 200)
        anno = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

        if view_mode == 1:
            left_img = anno
        else:
            left_img = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)

        # 等比缩放左侧图像
        scale = min(img_area_w / w, img_area_h / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)
        left_img = cv2.resize(left_img, (dw, dh))

        # 右侧：参数面板
        right_panel = build_panel(dh, g_h_min, g_h_max, s_h_min, s_h_max,
                                  erode, dilate, rate, green_area, soil_area,
                                  view_mode)

        # 左右拼接（如果高度不一致，取较大高度对齐）
        if left_img.shape[0] != right_panel.shape[0]:
            max_h = max(left_img.shape[0], right_panel.shape[0])
            if left_img.shape[0] < max_h:
                pad = np.zeros((max_h - left_img.shape[0], left_img.shape[1], 3), dtype=np.uint8)
                left_img = np.vstack([left_img, pad])
            if right_panel.shape[0] < max_h:
                pad = np.zeros((max_h - right_panel.shape[0], right_panel.shape[1], 3), dtype=np.uint8)
                right_panel = np.vstack([right_panel, pad])

        display = np.hstack([left_img, right_panel])

        cv2.imshow("Calibrate", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            view_mode = 1
        elif key == ord('2'):
            view_mode = 2
        elif key == ord('s'):
            print("\n=== 复制到 detector.py ===")
            print(f'GREEN_LOW  = np.array([{g_h_min}, {G_SAT[0]}, {G_VAL[0]}])')
            print(f'GREEN_HIGH = np.array([{g_h_max}, {G_SAT[1]}, {G_VAL[1]}])')
            print(f'SOIL_LOW   = np.array([{s_h_min}, {S_SAT[0]}, {S_VAL[0]}])')
            print(f'SOIL_HIGH  = np.array([{s_h_max}, {S_SAT[1]}, {S_VAL[1]}])')
            print(f'ERODE_K  = {erode}')
            print(f'DILATE_K = {dilate}')
            print("===========================\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
