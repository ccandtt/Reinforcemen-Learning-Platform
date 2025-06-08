from PySide6.QtCore import QObject, Signal
import numpy as np

class WorkerSignals(QObject):
    """Worker thread signals for training visualization."""
    
    # 环境相关信号
    env_frame_ready = Signal(np.ndarray)  # 迷宫环境帧
    
    # 训练进度信号
    progress_updated = Signal(int, int)    # 当前episode, 总episode数
    training_finished = Signal()           # 训练完成信号
    
    # 训练指标信号
    reward_updated = Signal(float, int)    # reward值, episode编号
    loss_updated = Signal(float, int)      # TD误差值, step编号
    
    # 状态信号
    state_updated = Signal(dict)           # 状态信息字典 