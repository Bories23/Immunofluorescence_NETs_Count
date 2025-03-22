#!/usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re
import PySimpleGUI as sg

# ----------------- 图像处理函数 -----------------

def extract_color_channel(image, channel_idx, color, threshold=50):
    """
    提取RGB单一通道，并转换为单色图像
    参数:
      threshold: 阈值，默认50
    """
    channel = image[:, :, channel_idx]
    binary_mask = channel > threshold
    output = np.zeros_like(image)
    output[binary_mask] = color
    return output, binary_mask.astype(np.uint8) * 255

def analyze_particles(mask, min_area):
    """
    分析色块并提取边缘
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(processed_mask, [contour], -1, 255, thickness=-1)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(processed_mask, kernel, iterations=1) - cv2.erode(processed_mask, kernel, iterations=1)
    return edges

def process_image_first(image_path, min_area, threshold=50, green_factor=100, red_factor=100, blue_factor=100, display_plots=True):
    """
    读取图像，分离RGB通道，分析色块并合并。
    参数:
      threshold: 用于 binary_mask 的阈值
      min_area: 色块最小面积阈值
      green_factor: 用于绿色通道：阈值为 min_area/green_factor
      red_factor: 用于红色通道：阈值为 min_area/red_factor
      blue_factor: 用于蓝色通道：阈值为 min_area/blue_factor
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    red_img, red_mask = extract_color_channel(image, 0, [255, 0, 0], threshold)
    green_img, green_mask = extract_color_channel(image, 1, [0, 255, 0], threshold)
    blue_img, blue_mask = extract_color_channel(image, 2, [0, 0, 255], threshold)

    red_edges = analyze_particles(red_mask, min_area / red_factor)
    green_edges = analyze_particles(green_mask, min_area / green_factor)
    blue_edges = analyze_particles(blue_mask, min_area / blue_factor)

    merged = np.zeros_like(image)
    merged[red_edges == 255] = [255, 0, 0]
    merged[green_edges == 255] = [0, 255, 0]
    merged[blue_edges == 255] = [0, 0, 255]

    if display_plots:
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 3, 1), plt.imshow(red_img), plt.title("Red Channel")
        plt.subplot(2, 3, 2), plt.imshow(green_img), plt.title("Green Channel")
        plt.subplot(2, 3, 3), plt.imshow(blue_img), plt.title("Blue Channel")
        plt.subplot(2, 3, 4), plt.imshow(red_edges, cmap="gray"), plt.title("Red Edges")
        plt.subplot(2, 3, 5), plt.imshow(green_edges, cmap="gray"), plt.title("Green Edges")
        plt.subplot(2, 3, 6), plt.imshow(blue_edges, cmap="gray"), plt.title("Blue Edges")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(merged)
        plt.title("Merged Image with Outlined Particles")
        plt.axis("off")
        plt.show()

    return merged

def process_image_second(image, display_plots=True):
    """
    使用第一步得到的合并图像（BGR格式）作为原始图像，
    转换为HSV颜色空间后进行颜色分割，
    对蓝色区域检测是否与红色和绿色区域接壤，
    如果同时接壤，则在蓝色轮廓上绘制白色边框。
    返回带有白色 label 的图像（BGR格式）和白色轮廓计数（white_count）。
    """
    labeled_image = image.copy()
    hsv = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2HSV)
    
    blue_lower = np.array([80, 100, 50])
    blue_upper = np.array([140, 255, 255])
    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 100, 50])
    green_upper = np.array([80, 255, 255])
    
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_count = 0
    for cnt in contours:
        blue_region_mask = np.zeros_like(blue_mask)
        cv2.drawContours(blue_region_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8)
        dilated_blue_region = cv2.dilate(blue_region_mask, kernel, iterations=1)
        overlap_red = cv2.bitwise_and(dilated_blue_region, red_mask)
        overlap_green = cv2.bitwise_and(dilated_blue_region, green_mask)
        red_overlap_count = np.count_nonzero(overlap_red)
        green_overlap_count = np.count_nonzero(overlap_green)
        if red_overlap_count >= 5 and green_overlap_count >= 5:
            cv2.drawContours(labeled_image, [cnt], -1, (255, 255, 255), thickness=2)
            white_count += 1

    print(f"Number of white-outlined shapes: {white_count}")
    if display_plots:
        labeled_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_rgb)
        plt.title("White Label Result")
        plt.axis("off")
        plt.show()

    return labeled_image, white_count

def numerical_key(filename):
    """
    根据文件名中的数字进行排序，没有数字则按原始名称排序。
    """
    base = os.path.basename(filename)
    nums = re.findall(r'\d+', base)
    if nums:
        return int(nums[0])
    else:
        return base

def batch_process_images(input_dir, output_dir, min_area=1000, threshold=50, green_factor=100, red_factor=100, blue_factor=100):
    """
    批量处理指定文件夹内的图片。
    对每张图片：
      1. 生成合并后的轮廓图，并保存到 processed 子文件夹；
      2. 进行白色 label 处理，并保存到 white_label 子文件夹；
      3. 返回每张图片白色轮廓的计数。
    图片按照名称（或文件名中的数字）顺序处理。
    """
    processed_dir = os.path.join(output_dir, "processed")
    white_label_dir = os.path.join(output_dir, "white_label")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(white_label_dir, exist_ok=True)
    
    exts = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        sg.popup("未找到符合条件的图片！")
        return []

    image_files = sorted(image_files, key=numerical_key)
    counts = []

    for image_path in image_files:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        print(f"正在处理: {base_name} ...")
        
        merged_rgb = process_image_first(image_path, min_area, threshold, green_factor, red_factor, blue_factor, display_plots=False)
        merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
        processed_image_path = os.path.join(processed_dir, f"{name}_processed{ext}")
        cv2.imwrite(processed_image_path, merged_bgr)
        
        labeled_image, white_count = process_image_second(merged_bgr, display_plots=False)
        white_label_path = os.path.join(white_label_dir, f"{name}_white_label{ext}")
        cv2.imwrite(white_label_path, labeled_image)
        
        counts.append(white_count)
        print(f"保存合并图: {processed_image_path}")
        print(f"保存白色 label 图: {white_label_path}")
    
    return counts

# ----------------- GUI 界面 -----------------

def main():
    sg.theme("LightBlue")
    layout = [
        [sg.Text("请选择包含图片的文件夹（支持拖拽文件夹）：")],
        [sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
        [sg.Text("输出文件夹（可为空，默认在输入文件夹下创建output_images）：")],
        [sg.Input(key="-OUTFOLDER-"), sg.FolderBrowse()],
        [sg.Text("Binary Threshold（默认50）："), sg.Input("50", key="-THRESH-")],
        [sg.Text("Min Area（默认1000）："), sg.Input("1000", key="-MINAREA-")],
        [sg.Text("Green Factor（默认100）："), sg.Input("100", key="-GREENFAC-")],
        [sg.Text("Red Factor（默认100）："), sg.Input("100", key="-REDFAC-")],
        [sg.Text("Blue Factor（默认100）："), sg.Input("100", key="-BLUEFAC-")],
        [sg.Button("开始处理"), sg.Button("退出")],
        [sg.Text("每张图片的白色轮廓计数（逗号分隔）：")],
        [sg.Multiline("", size=(50, 4), key="-RESULTS-")]
    ]
    
    window = sg.Window("批量图片处理程序", layout, size=(700, 450))
    
    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "退出"):
            break
        if event == "开始处理":
            input_folder = values["-FOLDER-"]
            output_folder = values["-OUTFOLDER-"]
            if not input_folder:
                sg.popup("请选择输入文件夹！")
                continue
            if not output_folder:
                output_folder = os.path.join(input_folder, "output_images")
            
            try:
                threshold = float(values["-THRESH-"])
                min_area = float(values["-MINAREA-"])
                green_factor = float(values["-GREENFAC-"])
                red_factor = float(values["-REDFAC-"])
                blue_factor = float(values["-BLUEFAC-"])
            except ValueError:
                sg.popup("请确保参数为数值！")
                continue

            try:
                counts = batch_process_images(input_folder, output_folder, min_area, threshold, green_factor, red_factor, blue_factor)
                counts_str = ",".join(str(c) for c in counts)
                window["-RESULTS-"].update(counts_str)
                sg.popup("批量处理完成！", f"处理了 {len(counts)} 张图片。")
            except Exception as e:
                sg.popup_error(f"处理过程中出错：{e}")
    
    window.close()

if __name__ == '__main__':
    main()
