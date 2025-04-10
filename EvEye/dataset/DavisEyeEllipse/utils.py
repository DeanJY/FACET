import cv2
import numpy as np


def cal_ellipse_area(major_axis, minor_axis):
    area = np.pi * major_axis * minor_axis / 4

    return area


def convert_to_ellipse(label):
    center = (float(label[1]), float(label[2]))
    axes = (float(label[3]), float(label[4]))
    angle = float(label[5])
    ellipse = (center, axes, angle)
    return ellipse


def get_input_size(keep_resolution, height, width):
    if keep_resolution:
        input_height = (height | 31) + 1
        input_width = (width | 31) + 1
        scale = np.array([input_width, input_height], dtype=np.float32)
    else:
        scale = max(height, width) * 1.0
        input_height, input_width = 256, 256
    return input_height, input_width, scale


def get_dir(src_point, rot_rad):  # 定义获取方向的函数
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)  # 计算旋转角度的正弦和余弦
    src_result = [0, 0]  # 初始化结果
    src_result[0] = src_point[0] * cs - src_point[1] * sn  # 计算 x 坐标
    src_result[1] = src_point[0] * sn + src_point[1] * cs  # 计算 y 坐标
    return src_result  # 返回计算结果


def get_3rd_point(a, b):  # 定义获取第三个点的函数
    direct = a - b  # 计算方向向量
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)  # 返回第三个点的坐标


def get_affine_transform(  # 定义获取仿射变换矩阵的函数
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(
        scale, list
    ):  # 检查 scale 类型
        scale = np.array([scale, scale], dtype=np.float32)  # 转换为数组

    scale_tmp = scale  # 临时保存 scale
    src_w = scale_tmp[0]  # 源宽度
    dst_w = output_size[0]  # 目标宽度
    dst_h = output_size[1]  # 目标高度

    rot_rad = np.pi * rot / 180  # 将角度转换为弧度
    src_dir = get_dir([0, src_w * -0.5], rot_rad)  # 获取源方向
    dst_dir = np.array([0, dst_w * -0.5], np.float32)  # 获取目标方向

    src = np.zeros((3, 2), dtype=np.float32)  # 初始化源点数组
    dst = np.zeros((3, 2), dtype=np.float32)  # 初始化目标点数组
    src[0, :] = center + scale_tmp * shift  # 计算源点
    src[1, :] = center + src_dir + scale_tmp * shift  # 计算源点方向
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # 计算目标点
    dst[1, :] = (
        np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    )  # 计算目标点方向

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])  # 计算源第三个点
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])  # 计算目标第三个点

    if inv:  # 如果需要逆变换
        trans = cv2.getAffineTransform(
            np.float32(dst), np.float32(src)
        )  # 获取逆仿射变换矩阵
    else:
        trans = cv2.getAffineTransform(
            np.float32(src), np.float32(dst)
        )  # 获取仿射变换矩阵
    return trans  # 返回变换矩阵


def gaussian2D(shape, sigma=1):  # 定义生成二维高斯分布的函数
    m, n = [(ss - 1.0) / 2.0 for ss in shape]  # 计算中心点
    y, x = np.ogrid[-m : m + 1, -n : n + 1]  # 生成网格坐标
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))  # 计算高斯值
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # 将小于阈值的高斯值设为 0
    return h  # 返回高斯分布


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def affine_transform(pt, t, angle=0, mode='xy'):  # 定义仿射变换函数
    if mode == 'xy':  # 如果模式为 'xy'
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T  # 将点转换为齐次坐标
        new_pt = np.dot(t, new_pt)  # 应用变换矩阵
    elif mode == 'ab':  # 如果模式为 'ab'
        angle = np.deg2rad(angle)  # 将角度转换为弧度
        cosA = np.abs(np.cos(angle))  # 计算余弦值
        sinA = np.abs(np.sin(angle))  # 计算正弦值
        a_x = pt[0] * cosA  # 计算 a 点的 x 坐标
        a_y = pt[0] * sinA  # 计算 a 点的 y 坐标
        b_x = pt[1] * sinA  # 计算 b 点的 x 坐标
        b_y = pt[1] * cosA  # 计算 b 点的 y 坐标
        new_pt_a = np.array([a_x, a_y, 0.0], dtype=np.float32).T  # a 点的齐次坐标
        new_pt_a = np.dot(t, new_pt_a)  # 应用变换矩阵
        new_pt_b = np.array([b_x, b_y, 0.0], dtype=np.float32).T  # b 点的齐次坐标
        new_pt_b = np.dot(t, new_pt_b)  # 应用变换矩阵
        new_pt = np.zeros((2, 1), dtype=np.float32)  # 初始化新点
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)  # 计算新点的 x 坐标
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)  # 计算新点的 y 坐标
    return new_pt[:2]  # 返回新点的前两个坐标


def gaussian_radius(det_size, min_overlap=0.7):  # 定义计算高斯半径的函数
    height, width = det_size  # 获取检测框的高度和宽度
    a1 = 1  # 一次方程的系数
    b1 = height + width  # 一次方程的常数项
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)  # 一次方程的常数项
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)  # 计算平方根
    r1 = (b1 + sq1) / 2  # 计算半径
    a2 = 4  # 二次方程的系数
    b2 = 2 * (height + width)  # 二次方程的常数项
    c2 = (1 - min_overlap) * width * height  # 二次方程的常数项
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)  # 计算平方根
    r2 = (b2 + sq2) / 2  # 计算半径
    a3 = 4 * min_overlap  # 三次方程的系数
    b3 = -2 * min_overlap * (height + width)  # 三次方程的常数项
    c3 = (min_overlap - 1) * width * height  # 三次方程的常数项
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)  # 计算平方根
    r3 = (b3 + sq3) / 2  # 计算半径
    return min(r1, r2, r3)  # 返回最小的半径
