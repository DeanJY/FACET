import cv2
import torch
import numpy as np

RESOLUTION = (64, 64)

def draw_ellipse(canvas, ellipse):
    x, y = ellipse[0]
    a, b = ellipse[1]
    an = ellipse[2]
    x = np.clip(x, 0, RESOLUTION[1] - 1)
    y = np.clip(y, 0, RESOLUTION[0] - 1)
    a = np.clip(a, 0, RESOLUTION[1] - 1)
    b = np.clip(b, 0, RESOLUTION[0] - 1)
    ellipse_to_draw = ((x, y), (a, b), an)
    cv2.ellipse(
        canvas,
        ellipse_to_draw,
        1,
        -1,
    )

def transform(images):
    batch_size = images.shape[0]
    canvases = []
    centers = []

    for i in range(batch_size):
        image = images[i].detach().cpu().numpy().astype(np.uint8)
        canvas = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours)>0 and len(contours[0])>=5:
            ellipse = cv2.fitEllipse(contours[0])
            a, b = ellipse[1]
            if a > 0 and b > 0:
                draw_ellipse(canvas, ellipse)
                canvas = cv2.resize(canvas, (RESOLUTION[0], RESOLUTION[1]))
                canvas = torch.from_numpy(canvas).float()
                center = torch.tensor((ellipse[0]), dtype=torch.float32)
                canvases.append(canvas)
                centers.append(center)
            else:
                canvases.append(torch.zeros(RESOLUTION[0], RESOLUTION[1]))
                centers.append(torch.zeros(2))
        else:
            canvases.append(torch.zeros(RESOLUTION[0], RESOLUTION[1]))
            centers.append(torch.zeros(2))
 
    canvas = torch.stack(canvases, dim=0)
    center = torch.stack(centers, dim=0)

    return {
        "canvas": canvas,
        "center": center,
    }


def p_acc(image, mask, tolerance=10):
    centerA = transform(image)["center"]
    centerB = transform(mask)["center"]

    if centerA.numel() == 0 or centerB.numel() == 0:
        return torch.tensor(float("nan"))

    distances = torch.norm(centerA - centerB, dim=1)
    overall_accuracy = (distances < tolerance).float().mean()

    return overall_accuracy


def cal_mean_distance(image, mask):
    centerA = transform(image)["center"]
    centerB = transform(mask)["center"]

    if centerA.numel() == 0 or centerB.numel() == 0:
        return torch.tensor(float("nan"))

    distances = torch.norm(centerA - centerB, dim=1)
    mean_distance = distances.mean()

    return mean_distance


def cal_batch_iou(image, mask):
    iou_list = []

    for i in range(image.shape[0]):
        canvas = image[i] + mask[i]
        inter = torch.sum(canvas == 2)
        union = torch.sum(canvas > 0)
        iou = 1.0 * inter / union if union > 0 else 0
        iou_list.append(iou)
    iou = torch.tensor(iou_list).float().mean()

    return iou
