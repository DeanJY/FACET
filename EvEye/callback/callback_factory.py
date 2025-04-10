from typing import Iterable, Type
import copy
from lightning.pytorch.callbacks import (
    ModelCheckpoint, 
    TQDMProgressBar,
    Timer,
    LearningRateMonitor,
)
from EvEye.callback.S3Checkpoint import S3Checkpoint
from lightning.pytorch.callbacks import Callback

CALLBACK_CLASSES: dict[str, Type[Callback]] = dict(
    ModelCheckpoint = ModelCheckpoint,
    TQDMProgressBar = TQDMProgressBar,
    Timer = Timer,
    LearningRateMonitor = LearningRateMonitor,
    S3Checkpoint = S3Checkpoint,
)

def make_callbacks(callback_cfgs: Iterable[dict] | dict) -> list[Callback]:
    callbacks = list()
    if isinstance(callback_cfgs, Iterable):
        callbacks = [make_single_callback(callback_cfg) for callback_cfg in callback_cfgs]
    else:
        callbacks = [make_single_callback(callback_cfgs)]

    return callbacks

def make_single_callback(callback_cfg: dict) -> Callback:
    assert callback_cfg['type'] in CALLBACK_CLASSES.keys()
    callback_cfg_ = copy.deepcopy(callback_cfg)
    return CALLBACK_CLASSES[callback_cfg_.pop('type')](**callback_cfg_)
