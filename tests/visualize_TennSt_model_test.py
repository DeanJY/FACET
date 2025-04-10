import torch
import cv2
import time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from EvEye.utils.dvs_common_utils.base.EventsIterator import EventsIterator
from EvEye.utils.processor.TxtProcessor import TxtProcessor
from EvEye.utils.dvs_common_utils.representation.TorchFrameStack import (
    TorchFrameStack,
)
from EvEye.utils.dvs_common_utils.representation.FrameStack import FrameStackBuilder
from EvEye.utils.dvs_common_utils.processor.EventRandomAffine import (
    EventRandomAffine,
    rand_range,
)
from EvEye.utils.visualization.visualization import *
from EvEye.dataset.DavisEyeCenter.losses import process_detector_prediction
from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model

start_time = time.time()

data_path = "/mnt/data2T/junyuan/Datasets/datasets/DavisEyeCenterDataset/data/user1_left_session_1_0_1_events.txt"
config_path = "TestTextDavisEyeDataset_TennSt.yaml"
output_path = "/mnt/data2T/junyuan/eye-tracking/outputs/TennSt_Output"


def streaming_inference(model, frames):
    model.eval()
    model.streaming()
    model.reset_memory()

    predictions = []
    with torch.inference_mode():
        for frame_id in range(frames.shape[2]):  # stream the frames to the model
            frame = frames[:, :, [frame_id]]
            prediction = model(frame)
            predictions.append(prediction)

    predictions = torch.cat(predictions, dim=2)
    return predictions


event = TxtProcessor(data_path).load_events_from_txt()
event["t"] = event["t"] - event["t"].min()
frame_gap_us = 40000
num_frames = event["t"].max() // frame_gap_us
start_time = event["t"].min()
event_iterator = EventsIterator(event, frame_gap_us, start_time)
output_path = Path(output_path)
output_path.mkdir(parents=True, exist_ok=True)
frame_stack_builder = FrameStackBuilder(height=260, width=346, num_frames=num_frames)
event = frame_stack_builder(
    xypt=event,
    start_time=start_time,
    spatial_downsample=(2, 2),
    temporal_downsample=1,
    mode="bilinear",
)
config = load_config(config_path)
model = make_model(config["model"])
model.load_state_dict(
    torch.load(config["test"]["ckpt_path"], map_location="cuda:0")["state_dict"]
)

predictions = []

event = torch.from_numpy(event).to(torch.float32)
# event = torch.randn(32, 2, 50, 96, 128)
event = event.permute(1, 0, 2, 3)
pred = streaming_inference(model, event[None, :])
pred = process_detector_prediction(pred)
pred = pred.squeeze(0)
predictions.append(pred)
print(event.shape)
predictions = torch.cat(predictions, dim=-1)
predictions[0] *= 346
predictions[1] *= 260
predictions_numpy = predictions.detach().numpy().T
predictions_numpy = np.concatenate(
    [np.arange(len(predictions_numpy))[:, None], predictions_numpy], axis=1
)

df = pd.DataFrame(predictions_numpy, columns=["row_id", "x", "y"])
df.to_csv("submission.csv", index=False)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f}s")
