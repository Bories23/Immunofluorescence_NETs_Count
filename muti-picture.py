import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

def extract_color_channel(image, channel_idx, color):
    """
    Extract a single RGB channel and convert it to a monochrome image.
    """
    channel = image[:, :, channel_idx]  # Get single channel
#----------------------------------------------------------------------------------------------------------------------------
    binary_mask = channel > ****  # Set threshold to extract specific color (default: 50)

    # Generate monochrome image
    output = np.zeros_like(image)
    output[binary_mask] = color  # Assign the corresponding color
    return output, binary_mask.astype(np.uint8) * 255  # Return the monochrome image and binary mask

def analyze_particles(mask, min_area):
    """
    Analyze colored blocks and extract edges.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Fill in the colored block
            cv2.drawContours(processed_mask, [contour], -1, 255, thickness=-1)

    # Get edges (dilation - erosion to obtain outlines)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(processed_mask, kernel, iterations=1) - cv2.erode(processed_mask, kernel, iterations=1)
    return edges

def process_image_first(image_path, min_area):
    """
    Read image, separate RGB channels, analyze colored blocks and merge,
    display each channel and the merged result. Returns the merged edge image (in RGB format).
    """
    # Read image and convert to RGB format
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Separate RGB colors and generate monochrome images
    red_img, red_mask = extract_color_channel(image, 0, [255, 0, 0])
    green_img, green_mask = extract_color_channel(image, 1, [0, 255, 0])
    blue_img, blue_mask = extract_color_channel(image, 2, [0, 0, 255])

    # Analyze colored blocks and extract edges
#----------------------------------------------------------------------------------------------------------------------------
    # Use a smaller minimum area factor for all channel (adjust as needed)
    red_edges = analyze_particles(red_mask, min_area / ****)
    green_edges = analyze_particles(green_mask, min_area / ****)
    blue_edges = analyze_particles(blue_mask, min_area / ****)

    # Create final merged image (edge map) in RGB format
    merged = np.zeros_like(image)
    merged[red_edges == 255] = [255, 0, 0]
    merged[green_edges == 255] = [0, 255, 0]
    merged[blue_edges == 255] = [0, 0, 255]

    # Display each channel and edge detection result (2x3 grid layout)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1), plt.imshow(red_img), plt.title("Red Channel")
    plt.subplot(2, 3, 2), plt.imshow(green_img), plt.title("Green Channel")
    plt.subplot(2, 3, 3), plt.imshow(blue_img), plt.title("Blue Channel")
    plt.subplot(2, 3, 4), plt.imshow(red_edges, cmap="gray"), plt.title("Red Edges")
    plt.subplot(2, 3, 5), plt.imshow(green_edges, cmap="gray"), plt.title("Green Edges")
    plt.subplot(2, 3, 6), plt.imshow(blue_edges, cmap="gray"), plt.title("Blue Edges")
    plt.tight_layout()
    plt.show()

    # Display merged image
    plt.figure(figsize=(6, 6))
    plt.imshow(merged)
    plt.title("Merged Image with Outlined Particles")
    plt.axis("off")
    plt.show()

    return merged

def process_image_second(image, show_image=True):
    """
    Use the merged image (in BGR format) from the first step as input,
    convert to HSV color space for color segmentation,
    check if blue regions are adjacent to both red and green regions,
    if adjacent to both, draw a white outline on the blue region edge,
    then display with matplotlib and save the result with white labels.

    Args:
      image: Input image in BGR format
      show_image: Whether to show the image with plt (set to False for batch processing)
    Returns:
      labeled_image: Image with white labels (in BGR format)
    """
    labeled_image = image.copy()
    hsv = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2HSV)

    # Define color threshold ranges
    blue_lower = np.array([80, 100, 50])
    blue_upper = np.array([140, 255, 255])

    red_lower1 = np.array([0, 100, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 100, 50])
    red_upper2 = np.array([180, 255, 255])

    green_lower = np.array([40, 100, 50])
    green_upper = np.array([80, 255, 255])

    # Generate color masks
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = red_mask1 | red_mask2
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find contours of blue regions
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    white_count = 0  # Count of white-outlined shapes

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

    labeled_rgb = cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)
    if show_image:
        plt.figure(figsize=(8, 6))
        plt.imshow(labeled_rgb)
        plt.title("White Label Result")
        plt.axis("off")
        plt.show()

    return labeled_image

def numerical_key(filename):
    """
    Sort based on numbers in the filename. If no numbers, sort by original name.
    """
    base = os.path.basename(filename)
    nums = re.findall(r'\d+', base)
    if nums:
        return int(nums[0])
    else:
        return base

def batch_process_images(input_dir, output_dir, min_area=1000):
    """
    Batch process images in the specified folder. For each image:
      1. Generate a merged contour image and save it to the 'processed' subfolder;
      2. Further process the result to add white labels and save it to the 'white_label' subfolder.
    Images are processed in order by filename (or numbers in the name).
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
        print("No matching images found.")
        return

    image_files = sorted(image_files, key=numerical_key)

    for image_path in image_files:
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        print(f"Processing: {base_name} ...")

        merged_rgb = process_image_first(image_path, min_area)
        merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
        processed_image_path = os.path.join(processed_dir, f"{name}_processed{ext}")
        cv2.imwrite(processed_image_path, merged_bgr)

        labeled_image = process_image_second(merged_bgr, show_image=False)
        white_label_path = os.path.join(white_label_dir, f"{name}_white_label{ext}")
        cv2.imwrite(white_label_path, labeled_image)

        print(f"Saved merged image: {processed_image_path}")
        print(f"Saved white label image: {white_label_path}")
    print("Batch processing completed.")

if __name__ == '__main__':
    # Single image processing example (for testing)
#----------------------------------------------------------------------------------------------------------------------------
    input_image_path = "****"  # Path to original image
    min_area = ****  # Minimum particle area threshold (pixels)
    merged_rgb = process_image_first(input_image_path, min_area)
    merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("processed_image.png", merged_bgr)
    process_image_second(merged_bgr)

    # Batch processing example: specify input and output folders (modify paths as needed)
#----------------------------------------------------------------------------------------------------------------------------
    input_dir = "****"   # Input image folder
    output_dir = "****"  # Output folder to save images
    batch_process_images(input_dir, output_dir, min_area)
