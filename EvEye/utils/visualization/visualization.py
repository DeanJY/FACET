import cv2
import torch
import os
import numpy as np


def visualize(
    frame_stack: np.array, threshold=0, normalized=False, add_weight: float = 0
):
    """
    Visualize the frame stack data in Numpy and matplotlib style (RGB).
    frame_stack: [T, P, H, W] or [P, H, W]
    """
    assert (
        frame_stack.ndim == 3 or frame_stack.ndim == 4
    ), "Invalid frame stack data shape."
    if frame_stack.ndim == 3:
        frame_stack = np.expand_dims(frame_stack, axis=0)
    t, p, h, w = frame_stack.shape
    if p == 2:
        canvases = np.zeros((t, h, w, 3), dtype=np.uint8)
        if normalized:
            for frame_index in range(t):
                off_channel = frame_stack[frame_index, 0]
                on_channel = frame_stack[frame_index, 1]
                canvases[frame_index, off_channel > threshold, 2] = (
                    (255 * (off_channel[off_channel > threshold] + add_weight))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                canvases[frame_index, on_channel > threshold, 0] = (
                    (255 * (on_channel[on_channel > threshold] + add_weight))
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        else:
            for frame_index in range(t):
                off_channel = frame_stack[frame_index, 0]
                on_channel = frame_stack[frame_index, 1]
                canvases[frame_index][off_channel > threshold] = [0, 0, 255]
                canvases[frame_index][on_channel > threshold] = [255, 0, 0]
    elif p == 1:
        canvases = np.zeros((t, h, w, 3), dtype=np.uint8)
        if normalized:
            for frame_index in range(t):
                polarity_channel = frame_stack[frame_index, 0]
                canvases[frame_index, polarity_channel < threshold, 2] = (
                    (
                        255
                        * (polarity_channel[polarity_channel < threshold] + add_weight)
                    )
                    .clip(0, 255)
                    .astype(np.uint8)
                )
                canvases[frame_index, polarity_channel > threshold, 0] = (
                    (
                        255
                        * (polarity_channel[polarity_channel > threshold] + add_weight)
                    )
                    .clip(0, 255)
                    .astype(np.uint8)
                )
        else:
            for frame_index in range(t):
                polarity_channel = frame_stack[frame_index, 0]
                canvases[frame_index][polarity_channel < threshold] = [0, 0, 255]
                canvases[frame_index][polarity_channel > threshold] = [255, 0, 0]
    if t == 1:
        return canvases[0]
    return canvases


def visualizeHWC(
    frame_stack: np.array, threshold=0, normalized=False, add_weight: float = 0
):
    frame_stack = np.transpose(frame_stack, (2, 0, 1))
    event_frame_vis = visualize(frame_stack, threshold, normalized, add_weight)

    return event_frame_vis


def load_image(
    image_path: str, color_mode: str = "color"  # image path  # 'color' or 'grayscale'
) -> tuple[np.ndarray, int, int]:
    """
    Load an image from the specified path and return the image array along with its width and height.

    Args:
        image_path (str): The path to the image file.
        color_mode (str): Color mode, supports 'color' (default, color) or 'grayscale'.

    Returns:
        tuple: A tuple containing the image array, width, and height of the image.
        Returns (None, 0, 0) if the image fails to load.
    """
    # Read the image according to the color mode
    if color_mode == "color":
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    elif color_mode == "grayscale":
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported color mode. Use 'color' or 'grayscale'.")

    # Check if the image is successfully loaded
    if image is None:
        print(
            f"Error: Unable to load the image from the path '{image_path}'. Please check if the path is correct."
        )
        return None, 0, 0

    height, width = image.shape[:2]

    return image, height, width


def save_image(
    image: np.ndarray, image_path: str, color_mode: str = "color", BGR2RGB: bool = True
) -> bool:
    """
    Save the image to the specified path with specified color modifications.

    Args:
        image (numpy.ndarray): The image array to be saved.
        image_path (str): The path to save the image, including the file name and extension.
        color_mode (str): "color" for color image, "grayscale" for grayscale image.
        BGR2RGB (bool): If True and image is color, convert from BGR to RGB format.

    Returns:
        bool: Whether the image is successfully saved.
    """
    if image is None or not isinstance(image, np.ndarray):
        # print("Invalid image data.")
        return False

    if color_mode == "grayscale":
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif color_mode != "color":
        print("Unsupported color mode. Use 'color' or 'grayscale'.")
        return False

    if BGR2RGB and image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return cv2.imwrite(image_path, image)


def save_batch_images(
    batch_images: np.array,
    images_path: str,
    start_index=0,
    color_mode: str = "color",
    BGR2RGB: bool = False,
):
    assert isinstance(batch_images, np.ndarray), "Invalid image data."
    assert batch_images.ndim == 4, "Invalid batch images data."
    for i, image in enumerate(batch_images, start=start_index):
        image_path = f"{images_path}_{i:02}.png"
        save_image(image, image_path, color_mode, BGR2RGB)


def resize_image(
    image: np.ndarray, size: tuple[int, int]  # 输入图像  # (height, width)
) -> tuple[np.ndarray, int, int]:
    """
    Resize the image to the specified size while maintaining the aspect ratio.

    Args:
        image (numpy.ndarray): The input image array.
        size (tuple): The target size (height, width) to resize the image.
    Returns:
        tuple: A tuple containing the resized image array, height, and width of the resized image.
    """
    raw_height, raw_width = image.shape[:2]
    target_height, target_width = size

    scale = min(target_width / raw_width, target_height / raw_height)

    new_height = int(raw_height * scale)
    new_width = int(raw_width * scale)

    # 使用cv2.resize调整图像大小
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )

    # 创建一个新图像，大小与目标大小相同，初始填充为灰色
    new_image = np.full((target_height, target_width, 3), 128, dtype=np.uint8)

    # 计算粘贴位置
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 将调整大小的图像粘贴到新图像上
    new_image[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_image
    )

    return new_image, new_height, new_width


def ensure_same_size(image1: np.array, image2: np.array) -> tuple[np.array, np.array]:
    """
    Ensure that two images have the same dimensions.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
    Returns:
        tuple: A tuple containing the two images with the same dimensions.
    """
    if image1.shape != image2.shape:
        raise ValueError(
            f"Image dimensions do not match: {image1.shape} vs {image2.shape}"
        )
    elif image1.shape == image2.shape:
        print(f"Image dimensions match: {image1.shape} vs {image2.shape}")
    return image1, image2


def ensure_same_dtype(image1: np.array, image2: np.array) -> tuple[np.array, np.array]:
    """
    Ensure that two images have the same data type.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
    Returns:
        tuple: A tuple containing the two images with the same data type.
    """
    if image1.dtype != image2.dtype:
        raise ValueError(
            f"Image data types do not match: {image1.dtype} vs {image2.dtype}"
        )
    elif image1.dtype == image2.dtype:
        print(f"Image data types match: {image1.dtype} vs {image2.dtype}")
    return image1, image2


def convert_to_color(*images):
    """
    Convert any number of grayscale images to color (RGB) images, returning them directly unpacked.

    Args:
        *images: Variable length image list where each image can be grayscale or already color.

    Returns:
        Single image or multiple images unpacked based on input. If one image is input, one image is returned without a list.
    """
    color_images = []
    for img in images:
        if len(img.shape) == 2 or (
            len(img.shape) == 3 and img.shape[2] == 1
        ):  # Check if the image is grayscale
            color_img = cv2.cvtColor(
                img, cv2.COLOR_GRAY2BGR
            )  # Convert grayscale to BGR
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            color_images.append(color_img)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # If already a color image
            color_images.append(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )  # Convert to RGB to standardize
        else:
            raise ValueError("Unsupported image format!")

    if len(color_images) == 1:
        return color_images[0]  # Return single image directly if only one is processed
    return tuple(color_images)


def get_color_map(num_classes: int = 5, type: int = 1) -> np.ndarray:
    """
    Generate a color map for a given number of classes.

    Args:
        num_classes (int): The number of classes.

    Returns:
        np.ndarray: An array of shape (num_classes, 3) containing RGB colors.
    """
    if type == 1:
        base_colors = [
            [0, 0, 0],  # Background color (black)
            [255, 0, 0],  # Class 1 color (red)
            [0, 255, 0],  # Class 2 color (green)
            [0, 0, 255],  # Class 3 color (blue)
            [255, 255, 0],  # Class 4 color (yellow)
            [255, 0, 255],  # Class 5 color (magenta)
            [0, 255, 255],  # Class 6 color (cyan)
            [255, 255, 255],  # Class 7 color (white)
        ]
    elif type == 2:
        base_colors = [
            [0, 0, 0],  # Background color (black)
            [128, 128, 0],  # Class 1 color (olive)
            [128, 0, 128],  # Class 2 color (purple)
            [0, 128, 128],  # Class 3 color (teal)
            [128, 0, 0],  # Class 3 color (maroon)
            [0, 128, 0],  # Class 4 color (green)
            [0, 0, 128],  # Class 5 color (navy)
            [128, 128, 128],  # Class 7 color (silver)
        ]
    # Extend the base_colors list to cover all classes
    while len(base_colors) < num_classes:
        base_colors += base_colors[: num_classes - len(base_colors)]

    # Ensure only the first num_classes colors are used
    color_map = np.array(base_colors[:num_classes], dtype=np.uint8)
    return color_map


def draw_points(canvas, points, color=(0, 255, 0), radius=0):
    """
    在给定的画布上绘制特征点。

    参数:
    canvas: 要绘制的画布
    points: 要绘制的特征点列表
    color: 点的颜色
    radius: 点的半径
    """

    canvas_to_draw = canvas.copy()
    for point in points:
        cv2.circle(canvas_to_draw, (int(point[0]), int(point[1])), radius, color, -1)

    return canvas_to_draw


def draw_label(label: np.ndarray, colormap_type: int = 1) -> np.ndarray:
    """
    Draw the colored image from the labels and color map.

    Args:
        label (np.ndarray): The labels as a 2D array where each element is a class index.
        colormap_type (int): The type of color map to use.

    Returns:
        np.ndarray: The colored image.
    """
    # Map labels to colors
    num_classes = len(np.unique(label))
    color_map = get_color_map(num_classes, colormap_type)
    colored_image = color_map[label]

    return colored_image


def overlay_label(
    image: np.ndarray,
    label: np.ndarray,
    colormap_type: int = 1,
    alpha: float = 0.5,
    ignore_background: bool = True,
) -> np.ndarray:
    """
    Overlay the colored labels on the image.

    Args:
        image (np.ndarray): The input image.
        label (np.ndarray): The labels as a 2D array where each element is a class index.
        colormap_type (int): The type of color map to use.
        alpha (float): The alpha value for blending, 1 represents full overlay.
        ignore_background (bool): Whether to ignore the background class (usually class 0).


    Returns:
        np.ndarray: The overlayed image.
    """
    image = image.copy()
    classes = np.unique(label)
    num_classes = len(np.unique(label))
    color_map = get_color_map(num_classes, colormap_type)
    # canvas = np.zeros_like(image, dtype=np.uint8)
    if ignore_background:
        for class_id in classes:
            if class_id == 0:
                continue
            mask = (label == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color_map[class_id].tolist(), -1)
    else:
        label = color_map[label.astype(int)]
        label = label.astype(image.dtype)
        image = cv2.addWeighted(image, 1 - alpha, label, alpha, 0)

    return image


def draw_contour(
    label: np.ndarray, line_thickness: int = 1, colormap_type: int = 1, image=None
) -> np.ndarray:
    """
    Draw the contours on the original labels image using the generated color map.

    Args:
        label (np.ndarray): The labels as a 2D array where each element is a class index.
        use_cv2 (bool): Whether to convert the output to BGR for OpenCV.
        line_thickness (int): Thickness of the contour lines.
        colormap_type (int): The type of color map to use.
    Returns:
        np.ndarray: The image with contours drawn.
    """
    num_classes = len(np.unique(label))
    color_map = get_color_map(num_classes, colormap_type)  # Get the color map
    if image is not None:
        image = image.copy()
        for i in range(1, num_classes):
            mask = (label == i).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color_map[i].tolist(), line_thickness)
        return image
    else:
        image_height, image_width = label.shape[:2]
        contour_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        for i in range(1, num_classes):  # Start from 1 to skip the background
            mask = (label == i).astype(
                np.uint8
            )  # Create a binary mask for the current class
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )  # Find contours
            cv2.drawContours(
                contour_image, contours, -1, color_map[i].tolist(), line_thickness
            )  # Draw contours

        return contour_image


def overlay_contour(
    image: np.array,
    label: np.array,
    line_thickness: int = 1,
    colormap_type: int = 1,
    ignore_background: bool = True,
    alpha: float = 0.5,
) -> np.array:
    """
    Overlay semi-transparent edges of labels directly onto the image ignoring background based on the flag.

    Args:
        image (np.ndarray): The original image in BGR format.
        label (np.ndarray): The labels as a 2D array where each element is a class index.
        line_thickness (int): The thickness of the contour lines to draw.
        alpha (float): The alpha value for blending, 1 represents full overlay.
        colormap_type (int): The type of color map to use.
        ignore_background (bool): Whether to ignore drawing contours for the background class (usually class 0).

    Returns:
        np.ndarray: The image with semi-transparent label edges drawn directly onto it.
    """
    image = image.copy()
    classes = np.unique(label)
    num_classes = len(np.unique(label))
    color_map = get_color_map(num_classes, colormap_type)
    canvas = np.zeros_like(image, dtype=np.uint8)
    if ignore_background:
        for class_id in classes:
            if class_id == 0:
                continue
            mask = (label == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image, contours, -1, color_map[class_id].tolist(), line_thickness
            )
    else:
        for class_id in classes:
            mask = (label == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                canvas, contours, -1, color_map[class_id].tolist(), line_thickness
            )
        image = cv2.addWeighted(image, 1 - alpha, canvas, alpha, 0)

    return image


def convert_to_ellipse(label):
    center = (float(label[1]), float(label[2]))
    axes = (float(label[3]), float(label[4]))
    angle = float(label[5])
    ellipse = (center, axes, angle)

    return ellipse


def draw_ellipse(canvas, ellipse, thickness=1):
    center = ellipse[0]
    axes = ellipse[1]
    angle = ellipse[2]

    long_axis_start = (
        int(center[0] - axes[0] / 2 * np.cos(np.radians(angle))),
        int(center[1] - axes[0] / 2 * np.sin(np.radians(angle))),
    )
    long_axis_end = (
        int(center[0] + axes[0] / 2 * np.cos(np.radians(angle))),
        int(center[1] + axes[0] / 2 * np.sin(np.radians(angle))),
    )

    cv2.line(canvas, long_axis_start, long_axis_end, (255, 255, 0), thickness)

    short_axis_start = (
        int(center[0] - axes[1] / 2 * np.sin(np.radians(angle))),
        int(center[1] + axes[1] / 2 * np.cos(np.radians(angle))),
    )
    short_axis_end = (
        int(center[0] + axes[1] / 2 * np.sin(np.radians(angle))),
        int(center[1] - axes[1] / 2 * np.cos(np.radians(angle))),
    )

    cv2.line(canvas, short_axis_start, short_axis_end, (255, 255, 0), thickness)

    cv2.circle(canvas, (int(center[0]), int(center[1])), thickness + 1, (0, 255, 0), -1)

    cv2.ellipse(canvas, ellipse, (0, 255, 0), thickness)
    # 打印椭圆参数，保留两位小数
    # print(
    #     f"椭圆中心: ({center[0]:.2f}, {center[1]:.2f}), 长轴: {axes[0]:.2f}, 短轴: {axes[1]:.2f}, 旋转角度: {angle:.2f}"
    # )


def main():
    image_path = "/mnt/data2T/junyuan/eye-tracking/datasets/Data_davis_labelled_with_mask/test/data/left_user10_session_1_0_2_0.png"
    label_path = "/mnt/data2T/junyuan/eye-tracking/outputs/TestDataset/left_user10_session_1_0_2_0.png"
    image = load_image(image_path, color_mode="color")[0]
    label = load_image(label_path, color_mode="grayscale")[0]
    _, label = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
    label[label == 255.0] = 1
    label = convert_to_color(label)
    ensure_same_dtype(image, label)
    ensure_same_size(image, label)
    result = overlay_label(
        image, label, alpha=0.1, colormap_type=1, ignore_background=False
    )
    save_image(result, "result.png", BGR2RGB=True)


if __name__ == "__main__":
    main()
