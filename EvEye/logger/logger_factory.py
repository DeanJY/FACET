from typing import Iterable  # 从 typing 模块导入 Iterable, 用于支持更丰富的类型注解
from lightning.pytorch.loggers import (
    Logger,
    TensorBoardLogger,
)  # 从 lightning 模块导入 Logger 基类和 TensorBoardLogger 类


# 定义函数 make_logger，根据传入的配置字典或字典列表生成对应的日志记录器列表
def make_logger(logger_cfgs: Iterable[dict] | dict) -> list[Logger]:
    loggers = list()  # 初始化空列表以存放日志记录器
    # 如果传入的配置是列表，逐个转换为日志记录器
    if isinstance(logger_cfgs, list):
        loggers = [make_single_logger(logger_cfg) for logger_cfg in logger_cfgs]
    # 如果传入的配置是单个字典，直接转换为单个日志记录器列表
    elif isinstance(logger_cfgs, dict):
        loggers = [make_single_logger(logger_cfgs)]
    return loggers  # 返回创建好的日志记录器列表


# 定义函数 make_single_logger，根据单个配置字典生成对应的日志记录器
def make_single_logger(logger_cfg: dict) -> Logger:
    # 检查配置中的类型是否为 "tensorboard"
    if logger_cfg["type"] == "tensorboard":
        # 根据配置创建 TensorBoardLogger 日志记录器，提供默认值以避免配置缺失
        logger = TensorBoardLogger(
            save_dir=logger_cfg.get("save_dir", "logs"),  # 日志保存目录，默认为 "logs"
            name=logger_cfg.get("name", "temp_exp"),  # 日志实验名，默认为 "temp_exp"
            version=logger_cfg.get("version"),  # 日志版本，无默认值，必须在配置中提供
        )
        return logger  # 返回创建的 TensorBoardLogger 实例
    else:
        # 如果类型不是 "tensorboard"，默认创建一个保存在 "logs" 目录的 TensorBoardLogger
        return TensorBoardLogger("logs")
