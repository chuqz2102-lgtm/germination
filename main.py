"""
Germination Check — 出芽率实时检测（左图右参 + 底部滑块可调）
"""

import cv2
import numpy as np
import os
import csv
import datetime
from detector import GREEN_LOW, GREEN_HIGH, SOIL_LOW, SOIL_HIGH

# ——— 配置 ———
WIN_W    = 1100
WIN_H    = 780    # 画面 ~650 + 滑块 ~130
PANEL_W  = 300
DATA_FILE = "germination_record.csv"

# S/V 固定值（不调）
G_S = (GREEN_LOW[1], GREEN_HIGH[1])
G_V = (GREEN_LOW[2], GREEN_HIGH[2])
S_S = (SOIL_LOW[1], SOIL_HIGH[1])
S_V = (SOIL_LOW[2], SOIL_HIGH[2])


def nothing(x):
    pass


def auto_white_balance(img):
    """Gray World 白平衡：让 RGB 三通道均值相等，去黄/蓝偏色"""
    b, g, r = cv2.split(img.astype(np.float32))
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    avg = (b_mean + g_mean + r_mean) / 3.0

    b = np.clip(b * (avg / max(b_mean, 1)), 0, 255).astype(np.uint8)
    g = np.clip(g * (avg / max(g_mean, 1)), 0, 255).astype(np.uint8)
    r = np.clip(r * (avg / max(r_mean, 1)), 0, 255).astype(np.uint8)

    return cv2.merge([b, g, r])


def record_data(rate, green_area, soil_area):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = not os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(["时间", "出芽率%", "绿色像素", "土壤像素"])
        w.writerow([now, f"{rate:.1f}", green_area, soil_area])


def load_history():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r") as f:
        lines = f.readlines()
    return lines[-6:] if len(lines) > 6 else lines[1:]


def build_panel(h_panel, rate, green_area, soil_area,
                g_h_min, g_h_max, s_h_min, s_h_max, e, d,
                view_mode, paused):
    panel = np.zeros((h_panel, PANEL_W, 3), dtype=np.uint8)
    panel[:, :] = (30, 30, 35)
    y = 15

    cv2.putText(panel, "LIVE DETECTION", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    if paused:
        cv2.putText(panel, "[PAUSED]", (180, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
    y += 40

    cv2.putText(panel, f"{rate:.1f}%", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    cv2.putText(panel, "Germination Rate", (10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 60

    cv2.putText(panel, f"Green: {green_area} px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(panel, f"Soil:  {soil_area} px", (10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 0), 1)
    y += 50

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    # 当前参数（实时读滑块）
    cv2.putText(panel, "PARAMS (drag sliders)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(panel, f"G H: {g_h_min}-{g_h_max}", (10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    cv2.putText(panel, f"G S: {G_S[0]}-{G_S[1]}  V: {G_V[0]}-{G_V[1]}", (10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
    cv2.putText(panel, f"S H: {s_h_min}-{s_h_max}", (10, y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 150, 0), 1)
    cv2.putText(panel, f"E={e}  D={d}", (150, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y += 80

    cv2.line(panel, (10, y), (PANEL_W - 10, y), (80, 80, 80), 1)
    y += 15

    cv2.putText(panel, "HISTORY", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    y += 22
    history = load_history()
    if history:
        for line in history[-5:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                t = parts[0][-8:]
                r = parts[1]
                cv2.putText(panel, f"{t}  {r}%", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 200, 180), 1)
                y += 18
    else:
        cv2.putText(panel, "(no records)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    cv2.putText(panel,
                f"View: {'Overlay' if view_mode == 1 else 'Green Mask'}  (1/2)",
                (10, h_panel - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    cv2.putText(panel, "c=capture  space=pause  s=save  q=quit",
                (10, h_panel - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    return panel


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    cv2.namedWindow("Germination Check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Germination Check", WIN_W, WIN_H)

    # 底部滑块
    cv2.createTrackbar("G_H_min", "Germination Check", GREEN_LOW[0],  179, nothing)
    cv2.createTrackbar("G_H_max", "Germination Check", GREEN_HIGH[0], 179, nothing)
    cv2.createTrackbar("S_H_min", "Germination Check", SOIL_LOW[0],   179, nothing)
    cv2.createTrackbar("S_H_max", "Germination Check", SOIL_HIGH[0],  179, nothing)
    cv2.createTrackbar("Erode",   "Germination Check", 0,             10,  nothing)
    cv2.createTrackbar("Dilate",  "Germination Check", 1,             10,  nothing)

    paused = False
    view_mode = 1
    last_frame = None
    saved_flash = 0

    img_area_w = WIN_W - PANEL_W
    img_area_h = WIN_H - 150  # 留 150px 给滑块

    print("=== Germination Check 出芽率实时检测 ===")
    print("  拖动底部滑块实时调参")
    print("  c    拍照记录")
    print("  s    保存当前参数到文件")
    print("  Space 暂停/继续")
    print("  1/2  切换视图")
    print("  q    退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头断连")
            break

        if not paused:
            last_frame = frame.copy()

        # 读滑块
        g_h_min = cv2.getTrackbarPos("G_H_min", "Germination Check")
        g_h_max = cv2.getTrackbarPos("G_H_max", "Germination Check")
        s_h_min = cv2.getTrackbarPos("S_H_min", "Germination Check")
        s_h_max = cv2.getTrackbarPos("S_H_max", "Germination Check")
        e_k     = max(cv2.getTrackbarPos("Erode",  "Germination Check"), 0)
        d_k     = max(cv2.getTrackbarPos("Dilate", "Germination Check"), 0)

        green_low  = np.array([g_h_min, G_S[0], G_V[0]])
        green_high = np.array([g_h_max, G_S[1], G_V[1]])
        soil_low   = np.array([s_h_min, S_S[0], S_V[0]])
        soil_high  = np.array([s_h_max, S_S[1], S_V[1]])

        # 图像处理
        hsv = cv2.cvtColor(last_frame, cv2.COLOR_BGR2HSV)
        mask_g = cv2.inRange(hsv, green_low, green_high)
        mask_s = cv2.inRange(hsv, soil_low, soil_high)

        if e_k > 0 or d_k > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_g = cv2.erode(mask_g, kernel, iterations=e_k)
            mask_g = cv2.dilate(mask_g, kernel, iterations=d_k)

        green_area = cv2.countNonZero(mask_g)
        # 从土壤 mask 中排除已被绿色覆盖的像素
        mask_s_only = cv2.bitwise_and(mask_s, cv2.bitwise_not(mask_g))
        soil_area = cv2.countNonZero(mask_s_only)
        total_area = green_area + soil_area
        rate = (green_area / total_area * 100) if total_area > 0 else 0.0

        # 标注图
        overlay = last_frame.copy()
        overlay[mask_g > 0] = (0, 255, 0)
        overlay[mask_s_only > 0] = (0, 120, 200)
        annotated = cv2.addWeighted(last_frame, 0.4, overlay, 0.6, 0)

        if view_mode == 1:
            left_img = annotated
        else:
            left_img = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)

        h_raw, w_raw = last_frame.shape[:2]
        scale = min(img_area_w / w_raw, img_area_h / h_raw, 1.0)
        dw, dh = int(w_raw * scale), int(h_raw * scale)
        left_img = cv2.resize(left_img, (dw, dh))

        right_panel = build_panel(dh, rate, green_area, soil_area,
                                  g_h_min, g_h_max, s_h_min, s_h_max,
                                  e_k, d_k, view_mode, paused)

        if left_img.shape[0] != right_panel.shape[0]:
            max_h = max(left_img.shape[0], right_panel.shape[0])
            if left_img.shape[0] < max_h:
                pad = np.zeros((max_h - left_img.shape[0], left_img.shape[1], 3), dtype=np.uint8)
                left_img = np.vstack([left_img, pad])
            if right_panel.shape[0] < max_h:
                pad = np.zeros((max_h - right_panel.shape[0], right_panel.shape[1], 3), dtype=np.uint8)
                right_panel = np.vstack([right_panel, pad])

        display = np.hstack([left_img, right_panel])

        if saved_flash > 0:
            cv2.putText(display, "SAVED!", (WIN_W // 2 - 80, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            saved_flash -= 1

        cv2.imshow("Germination Check", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            record_data(rate, green_area, soil_area)
            saved_flash = 15
            print(f"📸 记录: 出芽率 {rate:.1f}%")
        elif key == ord('s'):
            print("\n=== 保存到 detector.py ===")
            print(f'GREEN_LOW  = np.array([{g_h_min}, {G_S[0]}, {G_V[0]}])')
            print(f'GREEN_HIGH = np.array([{g_h_max}, {G_S[1]}, {G_V[1]}])')
            print(f'SOIL_LOW   = np.array([{s_h_min}, {S_S[0]}, {S_V[0]}])')
            print(f'SOIL_HIGH  = np.array([{s_h_max}, {S_S[1]}, {S_V[1]}])')
            print(f'ERODE_K  = {e_k}')
            print(f'DILATE_K = {d_k}')
            print("===========================\n")
        elif key == ord('1'):
            view_mode = 1
        elif key == ord('2'):
            view_mode = 2
        elif key == ord(' '):
            paused = not paused
            print("⏸  暂停" if paused else "▶  继续")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
