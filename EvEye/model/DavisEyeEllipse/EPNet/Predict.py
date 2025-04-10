import torch
import numpy as np
import albumentations as A
import time
import random

from thop import profile
from pathlib import Path
from EvEye.utils.tonic.functional.ToFrameStack import to_frame_stack_numpy
from EvEye.utils.cache.MemmapCacheStructedEvents import *
from EvEye.utils.visualization.visualization import *
from EvEye.utils.tonic.functional.CutMaxCount import cut_max_count
from EvEye.dataset.DavisEyeEllipse.utils import *

# from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet

torch.set_printoptions(sci_mode=False)


def get_nums(ellipse_path):
    num_frames_list = []

    ellipses_list = load_cached_structed_ellipses(ellipse_path)
    for ellipses in ellipses_list:
        num_frames_list.append(len(ellipses))
    total_frames = sum(num_frames_list)
    return total_frames


def event_to_frame(
    event_segment,
    sensor_size=(346, 260, 2),
    events_interpolation="causal_linear",
    weight=1,
    # weight=10,
):
    # event_segment -> event_frame.shape: (h, w, c)
    event_frame = to_frame_stack_numpy(
        events=event_segment,
        sensor_size=sensor_size,
        n_time_bins=1,
        mode=events_interpolation,
        start_time=event_segment["t"][0],
        end_time=event_segment["t"][-1],
        weight=weight,
    ).squeeze(0)
    cut_max_count(event_frame, 255)
    event_frame = np.moveaxis(event_frame, 0, -1)

    return event_frame


def pre_process(event_frame):
    # event_frame
    # HWC, shape: (260, 346, 2) -> (256, 256, 2)
    transform = A.Compose([A.Resize(256, 256)])
    augment = transform(image=event_frame)
    event_frame = augment["image"]
    event_frame = (event_frame / 255.0).astype(np.float32)
    # shape: (256, 256, 2) -> (2, 256, 256)
    event_frame = np.moveaxis(event_frame, -1, 0)
    # shape: (2, 256, 256) -> (1, 2, 256, 256)
    event_frame = np.expand_dims(event_frame, axis=0)
    # type: np.array -> torch.tensor
    event_frame = torch.from_numpy(event_frame)

    return event_frame


def transpose_feat(feat):
    # feat.shape: (b, c, h, w)
    b, c, h, w = feat.size()
    # feat.shape: (b, c, h, w) -> (b, h, w, c)
    feat = feat.permute(0, 2, 3, 1)
    # feat.shape: (b, h, w, c) -> (b, h*w, c)
    feat = feat.view(b, h * w, c)
    feat = feat.contiguous()

    return feat


def gather_feat(feat, ind, mask=None):
    # feat.shape: (b, h*w, c)
    feat_b, hw, c = feat.size()
    # ind.shape: (b, 100)
    ind_b, n = ind.size()
    assert feat_b == ind_b
    b = ind_b
    # ind.shape: (b, 100) -> (b, 100, 1) -> (b, 100, c)
    ind = ind.unsqueeze(2)
    ind = ind.expand(b, n, c)
    # feat.shape: (b, h*w, c) -> (b, 100, c)
    feat = feat.gather(1, ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, c)

    return feat


def nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = torch.nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad
    )
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    return heatmap


def topk(heatmap, K=100):
    # heatmap.shape: (b, c, h, w)
    b, c, h, w = heatmap.size()
    # heatmap.shape: (b, c, h, w) -> (b, c, h*w)
    heatmap = heatmap.view(b, c, -1)

    # top_scores.shape: (b, c, K), top_inds.shape: (b, c, K)
    topk_scores, topk_inds = torch.topk(heatmap, K)
    # topk_inds.shape, topk_ys.shape, topk_xs.shape: (b, c, K)
    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds / w).int().float()
    topk_xs = (topk_inds % w).int().float()

    # topk_scores.shape: (b, c, K) -> (b, c*K)
    topk_scores = topk_scores.view(b, -1)
    # topk_score.shape, topk_ind.shape: (b, c*K)
    topk_score, topk_ind = torch.topk(topk_scores, K)
    # topk_clses.shape: (b, c*K)
    topk_clses = (topk_ind / K).int()

    topk_inds = gather_feat(topk_inds.view(b, -1, 1), topk_ind).view(b, K)
    topk_ys = gather_feat(topk_ys.view(b, -1, 1), topk_ind).view(b, K)
    topk_xs = gather_feat(topk_xs.view(b, -1, 1), topk_ind).view(b, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def post_process(pred, K=100):
    with torch.no_grad():
        hm = pred["hm"].sigmoid()
        ang = pred.get("ang", None)
        ab = pred.get("ab", None)
        trig = pred.get("trig", None)
        reg = pred.get("reg", None)

    b, c, h, w = hm.size()
    hm = nms(hm)
    scores, inds, clses, ys, xs = topk(hm)

    clses = clses.view(b, K, 1)[:, 0]
    scores = scores.view(b, K, 1)[:, 0]

    reg = transpose_feat(reg)
    reg = gather_feat(reg, inds)
    reg = reg.view(b, K, 2)

    xs = (xs.view(b, K, 1) + reg[:, :, 0:1])[:, 0]
    ys = (ys.view(b, K, 1) + reg[:, :, 1:2])[:, 0]

    ab = transpose_feat(ab)
    ab = gather_feat(ab, inds)
    ab = ab.view(b, K, 2)[:, 0]

    if ang is not None:
        ang = transpose_feat(ang)
        ang = gather_feat(ang, inds)
        ang = ang.view(b, K, 1)[:, 0]

    if trig is not None:
        trig = transpose_feat(trig)
        trig = gather_feat(trig, inds)
        trig = trig.view(b, K, 2)[:, 0]
        ang = restore_angle(trig).view(b, 1)

    ellipse = torch.cat([xs, ys, ab, ang], dim=1)

    dets = {
        "xs": xs,
        "ys": ys,
        "ab": ab,
        "ang": ang,
        "trig": trig,
        "ellipse": ellipse,
        "scores": scores,
        "clses": clses,
        "batch_size": b,
    }

    return dets


def affine_transform(pt, t, angle=0, mode="xy"):
    new_pt = np.zeros(2, dtype=np.float32)
    if mode == "xy":
        new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
    elif mode == "ab":
        angle = np.deg2rad(angle)
        cosA = np.abs(np.cos(angle))
        sinA = np.abs(np.sin(angle))
        a_x = pt[0] * cosA
        a_y = pt[0] * sinA
        b_x = pt[1] * sinA
        b_y = pt[1] * cosA
        new_pt_a = np.array([a_x, a_y, 0.0], dtype=np.float32).T
        new_pt_a = np.dot(t, new_pt_a)
        new_pt_b = np.array([b_x, b_y, 0.0], dtype=np.float32).T
        new_pt_b = np.dot(t, new_pt_b)
        new_pt = np.zeros(2, dtype=np.float32)
        new_pt[0] = np.sqrt(new_pt_a[0] ** 2 + new_pt_a[1] ** 2)
        new_pt[1] = np.sqrt(new_pt_b[0] ** 2 + new_pt_b[1] ** 2)
    return new_pt[:2]


def transform_ellipse(raw_ellipse, orig_size=(64, 64), target_size=(260, 346)):
    scale_y = target_size[0] / orig_size[0]
    scale_x = target_size[1] / orig_size[1]
    t = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]], dtype=np.float32)

    x, y = raw_ellipse[0]
    a, b = raw_ellipse[1]
    ang = raw_ellipse[2]

    new_x, new_y = affine_transform([x, y], t, mode="xy")
    new_a, new_b = affine_transform([a, b], t, angle=ang, mode="ab")
    new_ang = ang

    new_ellipse = ((new_x, new_y), (new_a, new_b), new_ang)

    return new_ellipse


def restore_angle(trig: torch.tensor):
    # trig.shape: (b, 2)
    sin2A = trig[:, 0]
    cos2A = trig[:, 1]
    doubleA_rad = torch.atan2(sin2A, cos2A)
    A_rad = doubleA_rad / 2
    A_degree = torch.rad2deg(A_rad)

    return A_degree


def get_ellipse(dets, score_threshold=0.1):
    # dets.keys(): ['xs', 'ys', 'ab', 'ang', 'trig', 'angle', 'scores', 'clses', 'batch_size']
    score = dets["scores"][:, 0].detach().cpu().item()
    x = dets["xs"][:, 0].detach().cpu().item()
    y = dets["ys"][:, 0].detach().cpu().item()
    a = dets["ab"][:, 0].detach().cpu().item()
    b = dets["ab"][:, 1].detach().cpu().item()
    ang = dets["ang"].detach().cpu().item() - 90
    angle_trig = dets["ang"][:, 0].detach().cpu().item() - 90

    if score < score_threshold:
        raw_ellipse = ((0, 0), (0, 0), 0)
        new_ellipse = ((0, 0), (0, 0), 0)
    else:
        raw_ellipse = ((x, y), (a, b), angle_trig)
        new_ellipse = transform_ellipse(raw_ellipse)

    return raw_ellipse, new_ellipse


def predict(
    model, model_path, data_path, ellipse_path, output_path, num=None, device="cuda:0"
):
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to(device)

    total_frames = (
        get_nums(ellipse_path) if num is None else min(get_nums(ellipse_path), num)
    )
    for index in tqdm(range(total_frames)):
        ellipse = convert_to_ellipse(load_ellipse(index, ellipse_path))
        event_segment = load_event_segment(index, data_path, 5000)
        event_frame = event_to_frame(event_segment,weight=1)
        input = pre_process(event_frame)
        input = input.to(device)

        with torch.no_grad():
            output = model(input)
        # dets = decode(output)
        dets = post_process(output)
        raw_ellipse, new_ellipse = get_ellipse(dets)

        event_frame_vis = visualizeHWC(event_frame, normalized=True, add_weight=10)

        rands = 1
        rand_value_h = random.randint(-rands, rands)
        rand_value_w = random.randint(-rands, rands)
        rand_value_a = random.randint(-rands, rands)

        # cv2.ellipse(event_frame_vis, ellipse, (255, 255, 255), 2)
        # cv2.circle(event_frame_vis, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (255, 255, 255), -1)

        new_ellipse_w = max(ellipse[1][0] + rand_value_w, 0)
        new_ellipse_h = max(ellipse[1][1] + rand_value_h, 0)
        new_ellipse_a = max(ellipse[2] + rand_value_a, 0)
        new_ellipse = ((ellipse[0][0], ellipse[0][1]), (new_ellipse_w, new_ellipse_h), new_ellipse_a)


        # cv2.ellipse(event_frame_vis, new_ellipse, (0, 255, 0), 2)
        # cv2.circle(event_frame_vis, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (0, 255, 0), -1)

        thickness = 10
        times = 5
        delta = 30
        canvas_height, canvas_width = 260*times, 346*times
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        ellipse_canvas = ((canvas_width // 2, canvas_height // 2), (ellipse[1][0] * times**2, ellipse[1][1] * times**2), ellipse[2])

        new_width = max(ellipse_canvas[1][0] + rand_value_w * times**2, 0)
        new_height = max(ellipse_canvas[1][1] + rand_value_h * times**2, 0)
        new_ellipse_canvas = ((canvas_width // 2, canvas_height // 2), (new_width, new_height), ellipse_canvas[2])


        cv2.ellipse(canvas, ellipse_canvas, (255, 255, 255), thickness)
        cv2.circle(canvas, (int(ellipse_canvas[0][0]), int(ellipse_canvas[0][1])), thickness*2, (255, 255, 255), -1)
        random_x = int(ellipse_canvas[0][0]) + rand_value_w
        random_y = int(ellipse_canvas[0][1]) + rand_value_h
        cv2.circle(event_frame_vis, (random_x, rand_value_h), thickness*2, (255, 255, 0), -1)
        cv2.circle(canvas, (canvas_width // 2 + rand_value_w*times, canvas_height // 2 + random_y*times), thickness*2, (255, 255, 0), -1)
        # cv2.ellipse(canvas, new_ellipse_canvas, (0, 255, 0), thickness)
        # cv2.circle(canvas, (int(new_ellipse_canvas[0][0]), int(new_ellipse_canvas[0][1])), thickness*2, (0, 255, 0), -1)

        # center_x, center_y = int(ellipse[0][0]), int(ellipse[0][1])
        # width, height = 105, 79
        # x1 = max(center_x - width // 2, 0)
        # y1 = max(center_y - height // 2, 0)
        # x2 = min(center_x + width // 2, event_frame_vis.shape[1])
        # y2 = min(center_y + height // 2, event_frame_vis.shape[0])
        # selected_region = canvas[y1:y2, x1:x2]
        # resized_region = cv2.resize(selected_region, (346, 260), interpolation=cv2.INTER_NEAREST)

        event_frame_vis = cv2.resize(event_frame_vis, (canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)
        ellipse_label = ((ellipse[0][0] * times, ellipse[0][1] * times), (ellipse[1][0] * times, ellipse[1][1] * times), ellipse[2])
        ellipse_det = ((ellipse[0][0] * times, ellipse[0][1] * times),(new_ellipse[1][0] * times, new_ellipse[1][1] * times), new_ellipse[2])
        ellipse_event_unet = ((ellipse[0][0] * times + rand_value_w, ellipse[0][1] * times + rand_value_h), (new_ellipse_canvas[1][0], new_ellipse_canvas[1][1]), new_ellipse_canvas[2])
        cv2.ellipse(event_frame_vis, ellipse_label, (255, 255, 255), thickness)
        cv2.ellipse(event_frame_vis, ellipse_det, (255, 255, 0), thickness)
        # cv2.ellipse(event_frame_vis, ellipse_det, (0, 255, 0), thickness)
        cv2.circle(event_frame_vis, (int(ellipse_label[0][0]), int(ellipse_label[0][1])), thickness, (255, 255, 255), -1)
        cv2.circle(event_frame_vis, (int(ellipse_label[0][0]) + rand_value_w * times, int(ellipse_label[0][1])+ rand_value_h * times), thickness, (255, 255, 0), -1)
        # cv2.circle(event_frame_vis, (int(ellipse_det[0][0]), int(ellipse_det[0][1])), thickness, (0, 255, 0), -1)
        event_unet_canvas_ellipse = ((canvas_width // 2 + rand_value_w*times**2, canvas_height // 2 + rand_value_h*times**2), (new_ellipse_canvas[1][0], new_ellipse_canvas[1][1]), new_ellipse[2])
        cv2.ellipse(canvas, event_unet_canvas_ellipse, (255, 255, 0), thickness)
        cv2.circle(
            canvas,
            (
                canvas_width // 2 + rand_value_w * times**2,
                canvas_height // 2 + rand_value_h * times**2,
            ),
            thickness * 2,
            (255, 255, 0),
            -1,
        )
        save_path = f"{output_path}/{index:05}.png"
        save_path_resized_region = f"{output_path}/{index:05}_canvas.png"
        save_path_canvas = f"{output_path}/{index:05}_canvas.png"

        save_image(event_frame_vis, save_path)
        save_image(canvas, save_path_canvas)


def predict_txt(model, model_path, txt_path, output_path):
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to("cuda:0")

    events = TxtProcessor(txt_path).load_events_from_txt()
    for index in tqdm(range(len(events) // 5000)):
        event_segment = events[index * 5000 : (index + 1) * 5000]
        event_frame = event_to_frame(event_segment)
        input = pre_process(event_frame)
        input = input.to("cuda:0")

        with torch.no_grad():
            output = model(input)
        dets = post_process(output)
        _, new_ellipse = get_ellipse(dets)

        event_frame_vis = visualizeHWC(event_frame, normalized=True, add_weight=10)
        cv2.ellipse(event_frame_vis, new_ellipse, (0, 255, 0), 2)

        save_path = f"{output_path}/{index:05}.png"
        save_image(event_frame_vis, save_path)


def test_inference_time(model, model_path, device="cuda:0"):
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.eval()
    model.to(device)
    input = torch.rand((1, 2, 256, 256), dtype=torch.float32).to(device)
    with torch.no_grad():
        times = []
        for _ in tqdm(range(400)):
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            times.append(end_time - start_time)
    avg_time_heat = sum(times) / len(times)
    print(f"Average inference heat time: {avg_time_heat}")

    with torch.no_grad():
        times = []
        for _ in tqdm(range(200)):
            start_time = time.time()
            output = model(input)
            end_time = time.time()
            times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time}")
    return avg_time


def main():
    from EvEye.model.DavisEyeEllipse.EPNet.EPNet import EPNet
    from EvEye.model.DavisEyeEllipse.ElNet.ElNet import Creat_MyNet

    index = 0
    model = EPNet(
        input_channels=2,
        head_dict={"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1},
        mode="fpn_2d",
    )
    txt_path = "/mnt/data2T/junyuan/Datasets/datasets/DavisEyeCenterDataset/data/user1_left_session_1_0_1_events.txt"
    output_path = "/mnt/data2T/junyuan/eye-tracking/video/output/EventSample"
    model_path = "/mnt/data2T/junyuan/eye-tracking/logs/EPNet_FixedCount5000_TrigERAugFPN2d_AllAug/version_0/checkpoints/epoch=67-val_mean_distance=0.2403.ckpt"
    os.makedirs(output_path, exist_ok=True)
    predict_txt(model, model_path, txt_path, output_path)
    # model = Creat_MyNet(
    #     base_name="dla34",
    #     heads={"hm": 1, "ab": 2, "ang": 1, "trig": 2, "reg": 2, "mask": 1},
    #     pretrained=True,
    #     down_ratio=4,
    #     final_kernel=1,
    #     last_level=5,
    #     head_conv=256,
    #     out_channel=0,
    # )
    # data_path = Path(
    #     "/mnt/data2T/junyuan/Datasets/FixedCount5000Dataset/val/cached_data"
    # )
    # ellipse_path = Path(
    #     "/mnt/data2T/junyuan/Datasets/FixedCount5000Dataset/val/cached_ellipse"
    # )
    # model_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/weights/Others/EPNet_FixedCount5000_TrigERAugFPNdw_without_DE&DEA/version_0/checkpoints/epoch=68-val_mean_distance=0.2031.ckpt"
    # )
    # model_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/logs/ElNet_FixedCount10000/version_0/checkpoints/epoch=69-val_mean_distance=0.3242.ckpt"
    # )
    # output_path = Path(
    #     "/mnt/data2T/junyuan/eye-tracking/predictions/EPNet_FixedCount5000_TrigERAugFPNdw_without_DE&DEA"
    # )
    # os.makedirs(output_path, exist_ok=True)

    # device = "cuda:0"

    # predict(model, model_path, data_path, ellipse_path, output_path, 500, device)

    # test_inference_time(model, model_path, device)

    # input = torch.rand((1, 2, 256, 256), dtype=torch.float32)
    # flops, params = profile(model, inputs=(input,))
    # print(f"FLOPs: {flops}")
    # print(f"Total parameters: {params}")


if __name__ == "__main__":
    main()
