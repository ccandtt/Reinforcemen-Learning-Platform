from PySide6.QtCore import Signal, QObject
from PySide6.QtGui import QPixmap, QImage
import numpy as np


class Cv2FrameSignal(QObject):
    """
    自定义信号类，用于传递 OpenCV 图像帧。
    """
    frame_ready = Signal(object)  # 传递 OpenCV 图像帧
    error = Signal(str)  # 传递错误信息

    def __init__(self):
        super().__init__()
        self.latest_frame = None

    def set_frame(self, frame: np.ndarray):
        """
        设置最新的 OpenCV 图像帧，并发出信号。

        Args:
            frame: OpenCV 图像帧
        """
        print("set_frame")
        self.latest_frame = frame
        # 转成 QPixmap 并发出信号
        pixmap = QPixmap.fromImage(
            QImage(frame.data, frame.shape[1], frame.shape[0],
                   frame.strides[0], QImage.Format_BGR888)
        )
        self.frame_ready.emit(pixmap)



pipe_frame = Cv2FrameSignal()  # 实例化信号对象