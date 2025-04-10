import cv2
import torch
import numpy as np

RESOLUTION = (64, 64)  # h, w


def p_acc(dets, center, close, tolerance=10):
    valid_mask = close == 0
    center_pred = torch.cat((dets['xs'], dets['ys']), dim=1)[valid_mask]
    center = center[valid_mask]

    if center_pred.numel() == 0 or center.numel() == 0:
        return torch.tensor(float('nan'))

    distances = torch.norm(center - center_pred, dim=1)
    overall_accuracy = (distances < tolerance).float().mean()

    return overall_accuracy


def cal_mean_distance(dets, center, close):
    valid_mask = close == 0
    center_pred = torch.cat((dets['xs'], dets['ys']), dim=1)[valid_mask]
    center = center[valid_mask]

    if center_pred.numel() == 0 or center.numel() == 0:
        return torch.tensor(float('nan'))

    distances = torch.norm(center - center_pred, dim=1)
    mean_distance = distances.mean()

    return mean_distance


def draw_ellipse(canvas, ellipse):
    x, y, a, b, an = ellipse
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


def cal_iou(ellipseA, ellipseB):
    canvasA = np.zeros((RESOLUTION[0], RESOLUTION[1]), dtype=np.uint8)
    canvasB = np.zeros((RESOLUTION[0], RESOLUTION[1]), dtype=np.uint8)

    if ellipseA[2] <= 0 or ellipseA[3] <= 0 or ellipseB[2] <= 0 or ellipseB[3] <= 0:
        return 0

    draw_ellipse(canvasA, ellipseA)
    draw_ellipse(canvasB, ellipseB)

    canvas = canvasA + canvasB
    inter = np.sum(canvas == 2)
    union = np.sum(canvas > 0)

    iou = 1.0 * inter / union if union > 0 else 0

    return iou


def cal_batch_iou(dets, ellipse, close):
    valid_mask = close == 0
    ellipse_pred = dets['ellipse'][valid_mask]
    ellipse = ellipse[valid_mask]

    iou_list = []

    for i in range(ellipse_pred.shape[0]):
        ellipse_pred_i = ellipse_pred[i].detach().cpu().numpy()
        ellipse_label_i = ellipse[i].detach().cpu().numpy()
        iou_i = cal_iou(ellipse_pred_i, ellipse_label_i)
        iou_list.append(iou_i)

    iou = torch.tensor(iou_list).float().mean()

    return iou


def cal_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def cal_batch_ap(dets, ellipse, close, iou_thres=0.5, score_threshold=0.5):
    valid_mask = close == 0
    ellipse_pred = dets['ellipse'][valid_mask]
    ellipse = ellipse[valid_mask]
    scores = dets['scores'][valid_mask]

    tp = []
    fp = []
    fn = []
    gt_num = ellipse.shape[0]

    for i in range(gt_num):
        if scores[i] < score_threshold:
            fn.append(1)
            tp.append(0)
            fp.append(0)
            continue

        ellipse_pred_i = ellipse_pred[i].detach().cpu().numpy()
        ellipse_label_i = ellipse[i].detach().cpu().numpy()

        iou = cal_iou(ellipse_pred_i, ellipse_label_i)
        if iou > iou_thres:
            tp.append(1)
            fp.append(0)
            fn.append(0)
        else:
            tp.append(0)
            fp.append(1)
            fn.append(0)

    tp = np.array(tp)
    fp = np.array(fp)
    fn = np.array(fn)
    scores = scores.detach().cpu().numpy()

    # Sort by score
    sorted_indices = np.argsort(-scores)
    tp = tp[sorted_indices]
    fp = fp[sorted_indices]
    fn = fn[sorted_indices]

    # Compute cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    fn_cumsum = np.cumsum(fn)

    # Compute precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / (tp_cumsum + fn_cumsum)

    # Compute AP
    ap = cal_ap(recall, precision)

    ap = torch.tensor(ap, dtype=torch.float32)
    recall = torch.tensor(recall, dtype=torch.float32)
    precision = torch.tensor(precision, dtype=torch.float32)

    return ap
