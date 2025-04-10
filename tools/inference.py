import torch
import numpy as np
import pandas as pd

from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.dataset_factory import make_dataset
from EvEye.model.model_factory import make_model
from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.DavisEyeCenter.losses import process_detector_prediction


def main():
    config_path = (
        "/mnt/data2T/junyuan/eye-tracking/configs/TestTextDavisEyeDataset_TennSt.yaml"
    )

    config = load_config(config_path)
    testDataset = make_dataset(config["dataset"])
    model = make_model(config["model"])
    model.load_state_dict(torch.load(config["test"]["ckpt_path"])["state_dict"])
    device = config["test"]["map_location"]
    model.to(device)

    event_frames = testDataset[0].unsqueeze(0).to(device)

    pred = model.streaming_inference(model, event_frames)
    pred = process_detector_prediction(pred)
    pred = pred.squeeze(0)
    pred[0] *= 346
    pred[1] *= 260
    predictions_numpy = pred.detach().cpu().numpy().T.astype(np.int32)

    arange = np.arange(predictions_numpy.shape[0])
    predictions_numpy = np.concatenate([arange[:, None], predictions_numpy], axis=1)

    df = pd.DataFrame(predictions_numpy, columns=["row_id", "x", "y"])
    df.to_csv("submission.csv", index=False)

    print("Inference completed.")


if __name__ == "__main__":
    main()
