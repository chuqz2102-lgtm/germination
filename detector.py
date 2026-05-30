"""
出芽率检测核心模块 — 可复制到主程序里直接调用
"""

import cv2
import numpy as np


# ========== 默认参数（用 calibrator.py 调好后替换这里）==========
# H: 0-179 (OpenCV), S: 0-255, V: 0-255
# 绿色：覆盖黄绿嫩芽(H~30)到深绿成熟叶(H~88)，低饱和/低亮度也纳入
GREEN_LOW  = np.array([30, 20, 40])
GREEN_HIGH = np.array([88, 200, 230])
# 土壤：收窄，排除阴影和杂质，只检测明确是土壤的像素
SOIL_LOW   = np.array([5, 15, 30])
SOIL_HIGH  = np.array([30, 220, 220])
ERODE_K  = 0
DILATE_K = 1


def calc_germination_rate(frame,
                          green_low=GREEN_LOW, green_high=GREEN_HIGH,
                          soil_low=SOIL_LOW, soil_high=SOIL_HIGH,
                          erode_k=ERODE_K, dilate_k=DILATE_K):
    """
    输入：BGR 图像帧（numpy array）
    返回：(出芽率%, 绿色面积px, 土壤面积px, 绿色掩膜, 标注图)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv, green_low, green_high)
    mask_soil  = cv2.inRange(hsv, soil_low, soil_high)

    # 形态学去噪
    if erode_k > 0 or dilate_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if erode_k > 0:
            mask_green = cv2.erode(mask_green, kernel, iterations=erode_k)
        if dilate_k > 0:
            mask_green = cv2.dilate(mask_green, kernel, iterations=dilate_k)

    green_area = cv2.countNonZero(mask_green)
    # 从土壤 mask 中排除已被绿色覆盖的像素，避免分母缩小
    mask_soil_only = cv2.bitwise_and(mask_soil, cv2.bitwise_not(mask_green))
    soil_area = cv2.countNonZero(mask_soil_only)
    total_area = green_area + soil_area

    # 正确公式：出芽率 = 绿色面积 / (绿色+土壤) × 100%
    # 这样即使绿色完全覆盖土壤，分母也不会缩小
    rate = (green_area / total_area * 100) if total_area > 0 else 0.0

    # 标注图
    overlay = frame.copy()
    overlay[mask_green > 0] = (0, 255, 0)
    overlay[mask_soil_only > 0]  = (0, 100, 200)
    annotated = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

    cv2.putText(annotated, f"Germination: {rate:.1f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Green: {green_area}px  Soil: {soil_area}px",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return rate, green_area, soil_area, mask_green, annotated


def calc_from_file(image_path, **kwargs):
    """从图片文件计算出芽率"""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return calc_germination_rate(img, **kwargs)


# ========== 独立运行 ==========
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python detector.py <图片路径>")
        sys.exit(1)

    rate, ga, sa, mask, anno = calc_from_file(sys.argv[1])
    print(f"出芽率: {rate:.1f}%  绿色={ga}px  土壤={sa}px")

    cv2.imshow("Result", anno)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
