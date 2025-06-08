import sys
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLineEdit,
    QLabel,
    QProgressBar,
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QPixmap, QImage
from utils.pipe_frame import pipe_frame  # 假设 pipe_frame.py 在 utils 目录下
import cv2
import numpy as np
from config import PARAMS_CONFIG  # 定义了不同场景及算法 --参数


# # 模拟功能函数 API
# def run_algorithm(scene, algorithm, params):
#     """
#     模拟强化学习算法 API，接受场景、算法和参数字典，返回一张图片。

#     Args:
#         scene (str): 选择的场景名称
#         algorithm (str): 选择的算法名称
#         params (dict): 参数字典

#     Returns:
#         QPixmap: 返回的图片对象
#     """
#     from main import main_SARSA, main_DQN, main_A2C, main_DDPG, main_PPO
#     print(f"Running {algorithm} on {scene} with parameters: {params}")
#     config = params
#     if config['algorithm'] == 'SARSA':
#         main_SARSA(config)
    # elif config['algorithm'] == 'DQN':
    #     main_DQN(config)
    # elif config['algorithm'] == 'A2C':
    #     main_A2C(config)
    # elif config['algorithm'] == 'DDPG':
    #     main_DDPG(config)
    # elif config['algorithm'] == 'PPO':
    #     main_PPO(config)


# 核心：开始创建某个算法 --调用main_DDPG.py
class WorkerThread(QThread):
    def __init__(self, scene, algorithm, params, main_window):
        super().__init__()
        self.scene = scene
        self.algorithm = algorithm
        self.params = params
        self.main_window = main_window

    def run(self):
        if self.scene == 'Pendulum-v1':
            if self.algorithm == 'DDPG':
                from main_DDPG import main_DDPG
                trainer = main_DDPG(self.params, return_trainer=True)
                
                # 连接训练器信号
                trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
                trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
                trainer.signals.progress_updated.connect(self.main_window.update_progress)
                
                # 开始训练
                trainer.train()

        elif self.scene == 'CartPole-v0':
            if self.algorithm == 'DQN':
                from main_DQN import main_DQN
                trainer = main_DQN(self.params, return_trainer=True)
            elif self.algorithm == 'A2C':
                from main_A2C import main_A2C
                trainer = main_A2C(self.params, return_trainer=True)
            elif self.algorithm == 'PPO':
                from main_PPO import main_PPO
                trainer = main_PPO(self.params, return_trainer=True)
                
            # 连接训练器信号
            trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
            trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
            trainer.signals.progress_updated.connect(self.main_window.update_progress)
            
            # 开始训练
            if self.algorithm == 'DQN':
                trainer.train("time_penalty", self.params.get('td_target', False))
            elif self.algorithm == 'A2C':
                trainer.train("time_penalty", "CartPole-v0")
            elif self.algorithm == 'PPO':
                trainer.train("time_penalty", "CartPole-v0")
                
        elif self.scene == '迷宫':
            if self.algorithm == 'DQN':
                from main_DQN import main_DQN
                trainer = main_DQN(self.params, return_trainer=True)
                
                # 确保参数中包含迷宫相关配置
                self.params.update({
                    'env_name': 'maze-v0',
                    'maze_size': self.params.get('maze_size', 5),
                    'mode': 'rgb_array'  # 确保环境返回RGB数组
                })
                
                # 连接信号
                trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
                trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
                trainer.signals.progress_updated.connect(self.main_window.update_progress)
                
                # 开始训练
                trainer.train("maze", False)  # 使用maze环境特定的奖励类型
            elif self.algorithm == 'SARSA':
                from main_SARSA import main_SARSA
                trainer = main_SARSA(self.params, return_trainer=True)
                # 连接信号
                trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
                trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
                trainer.signals.progress_updated.connect(self.main_window.update_progress)
                # 开始训练
                trainer.train("time_penalty", self.params.get('td_target', False))
            # 你可以继续加 elif self.algorithm == 'A2C' 等
            elif self.algorithm == 'A2C':
                from main_A2C import main_A2C
                trainer = main_A2C(self.params, return_trainer=True)
                # 连接信号
                trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
                trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
                trainer.signals.progress_updated.connect(self.main_window.update_progress)
                # 开始训练
                trainer.train("time_penalty", self.params.get('td_target', False))
            elif self.algorithm == 'PPO':
                from main_PPO import main_PPO
                trainer = main_PPO(self.params, return_trainer=True)
                # 连接信号
                trainer.signals.env_frame_ready.connect(self.main_window.update_env_view)
                trainer.signals.plot_frame_ready.connect(self.main_window.update_plot_view)
                trainer.signals.progress_updated.connect(self.main_window.update_progress)
                # 开始训练
                trainer.train("time_penalty", self.params.get('td_target', False))

  
        else:
            result = run_algorithm(self.scene, self.algorithm, self.params)



# QSS 浅色样式
QSS_STYLE = """
QMainWindow {
    background-color: #F5F5F5;
}
QWidget#sceneWidget, QWidget#mainWidget {
    background-color: #F5F5F5;
}
QPushButton {
    background-color: #2196F3;
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-size: 16px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #1976D2;
}
QPushButton:pressed {
    background-color: #1565C0;
}
QComboBox {
    background-color: #FFFFFF;
    color: #333333;
    border: 1px solid #B0BEC5;
    border-radius: 5px;
    padding: 5px;
    font-size: 14px;
}
QComboBox::drop-down {
    border: none;
}
QLabel#titleLabel {
    color: #333333;
    font-size: 24px;
    font-weight: bold;
}
QLabel {
    color: #333333;
    font-size: 14px;
}
QSpinBox, QDoubleSpinBox {
    background-color: #FFFFFF;
    color: #333333;
    border: 1px solid #B0BEC5;
    border-radius: 5px;
    padding: 5px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #BBDEFB;
    border: none;
    border-radius: 3px;
    width: 20px;
    height: 15px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #BBDEFB;
    border: none;
    border-radius: 3px;
    width: 20px;
    height: 15px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: transparent;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: transparent;
}
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background-color: transparent;
}
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background-color: transparent;
}
QCheckBox {
    color: #333333;
    font-size: 14px;
}
QLineEdit {
    background-color: #FFFFFF;
    color: #333333;
    border: 1px solid #B0BEC5;
    border-radius: 5px;
    padding: 5px;
}
QWidget#imageWidget {
    background-color: #FFFFFF;
    border: 1px solid #B0BEC5;
    border-radius: 10px;
}

/* 隐藏步进按钮的样式（取消注释以使用） */
/*
QSpinBox::up-button, QDoubleSpinBox::up-button {
    width: 0px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    width: 0px;
}
*/
"""


class ParameterWidgetFactory:
    """参数控件工厂，动态生成不同类型的参数输入控件"""

    @staticmethod
    def create_widget(param_name, param_config):
        widget_type = param_config["type"]
        if widget_type == "int":
            widget = QSpinBox()
            widget.setMinimum(param_config["min"])
            widget.setMaximum(param_config["max"])
            widget.setSingleStep(param_config["step"])
            widget.setValue(param_config["value"])
            return param_name, widget
        elif widget_type == "float":
            widget = QDoubleSpinBox()
            widget.setMinimum(param_config["min"])
            widget.setMaximum(param_config["max"])
            widget.setSingleStep(param_config["step"])
            widget.setValue(param_config["value"])
            return param_name, widget
        elif widget_type == "bool":
            widget = QCheckBox(param_name)
            widget.setChecked(param_config["value"])
            return param_name, widget
        elif widget_type == "string":
            widget = QLineEdit()
            widget.setText(param_config["value"])
            return param_name, widget
        elif widget_type == "list":
            widget = QLineEdit()
            # Convert list to string representation
            widget.setText(str(param_config["value"]))
            return param_name, widget
        return None, None


class SceneSelectionWindow(QWidget):
    """场景选择窗口，使用信号传递选中的场景"""

    scene_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sceneWidget")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("选择环境")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333333;")
        layout.addWidget(title)

        for scene in ["迷宫", "Pendulum-v1", "CartPole-v0"]:
            button = QPushButton(scene)
            button.clicked.connect(lambda checked, s=scene: self.scene_selected.emit(s))
            layout.addWidget(button)

        layout.addStretch()
        self.setLayout(layout)


class MainOperationWindow(QWidget):
    """主操作界面"""
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setObjectName("mainWidget")
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        
        # 标题栏
        title = QLabel(f"{self.scene} - 算法配置")
        title.setObjectName("titleLabel")
        main_layout.addWidget(title)
        
        # 主体部分使用水平布局
        content_layout = QHBoxLayout()
        
        # 左侧：算法选择和参数配置
        left_widget = QWidget()
        left_widget.setFixedWidth(300)  # 固定宽度
        left_layout = QVBoxLayout()
        
        # 算法选择下拉框
        self.algorithm_combo = QComboBox()
        algorithms = list(PARAMS_CONFIG[self.scene].keys())
        self.algorithm_combo.addItems(algorithms)
        self.algorithm_combo.currentTextChanged.connect(self.update_parameters)
        left_layout.addWidget(QLabel("选择算法:"))
        left_layout.addWidget(self.algorithm_combo)
        
        # 参数配置区域
        self.params_layout = QFormLayout()
        left_layout.addLayout(self.params_layout)
        
        # 进度显示
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        left_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        left_layout.addWidget(self.status_label)
        
        # 运行按钮
        run_button = QPushButton("运行")
        run_button.clicked.connect(self.run_algorithm)
        left_layout.addWidget(run_button)
        
        left_layout.addStretch()
        left_widget.setLayout(left_layout)
        content_layout.addWidget(left_widget)
        
        # 右侧：训练可视化（使用垂直布局）
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # 环境渲染（上半部分）
        env_container = QWidget()
        env_container.setMinimumHeight(300)
        env_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        env_layout = QVBoxLayout()
        
        self.env_view = QLabel()
        self.env_view.setAlignment(Qt.AlignCenter)
        self.env_view.setMinimumSize(500, 300)
        env_layout.addWidget(self.env_view)
        
        env_container.setLayout(env_layout)
        right_layout.addWidget(env_container)
        
        # 训练曲线（下半部分）
        plot_container = QWidget()
        plot_container.setMinimumHeight(300)
        plot_container.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        plot_layout = QVBoxLayout()
        
        self.plot_view = QLabel()
        self.plot_view.setAlignment(Qt.AlignCenter)
        self.plot_view.setMinimumSize(500, 300)
        plot_layout.addWidget(self.plot_view)
        
        plot_container.setLayout(plot_layout)
        right_layout.addWidget(plot_container)
        
        right_widget.setLayout(right_layout)
        content_layout.addWidget(right_widget)
        
        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)
        
        # 初始化参数
        self.update_parameters(self.algorithm_combo.currentText())
        
    def update_parameters(self, algorithm):
        """动态更新参数配置区域"""
        # 清空现有参数控件
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 生成新参数控件
        params = PARAMS_CONFIG[self.scene].get(algorithm, {})  # 加载 
        ''' 加载参数配置示例
         {
        "env_name": "Pendulum-v1",
        "reward_decay": 0.99,
        "learning_rate": 3e-4,
        "hide_n1": 3,
        "hide_list1": "[128,128,128]",
        "hide_n2": 3, 
        "hide_list2": "[128,128,128]",
        "episode": 100,
        "td_target": True,
        "n_step": 3,
        "use_per": True
        }
                
        '''
        for param_name, param_config in params.items():
            label, widget = ParameterWidgetFactory.create_widget(
                param_name, param_config
            )
            if widget:
                self.params_layout.addRow(label, widget)

    
    # 当用户点击"运行"按钮后：
    def run_algorithm(self):
        """收集参数并调用功能函数 API"""
        algorithm = self.algorithm_combo.currentText()
        params = {}
        
        # 获取当前算法的参数配置
        algorithm_params = PARAMS_CONFIG[self.scene][algorithm]
        
        for i in range(self.params_layout.rowCount()):
            label_item = self.params_layout.itemAt(i, QFormLayout.LabelRole)
            field_item = self.params_layout.itemAt(i, QFormLayout.FieldRole)
            if label_item and field_item:
                param_name = (
                    label_item.widget().text()
                    if isinstance(label_item.widget(), QLabel)
                    else label_item.widget().text()
                )
                widget = field_item.widget()
                
                # 根据参数类型处理值
                param_type = algorithm_params[param_name]["type"]
                
                if param_type == "int":
                    params[param_name] = widget.value()
                elif param_type == "float":
                    params[param_name] = widget.value()
                elif param_type == "bool":
                    params[param_name] = widget.isChecked()
                elif param_type == "string":
                    params[param_name] = widget.text()
                elif param_type == "list":
                    # 将字符串形式的列表转换为实际的列表
                    try:
                        params[param_name] = eval(widget.text())
                    except:
                        # 如果转换失败，使用默认值
                        params[param_name] = algorithm_params[param_name]["value"]
                        
        params["algorithm"] = algorithm

        # 停止之前的训练（如果存在）
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()  # 等待线程真正结束

        # 创建新的工作线程-- 创建某个实例
        self.worker = WorkerThread(self.scene, algorithm, params, self)
        
        # 重置进度条和状态
        self.progress_bar.setValue(0)
        self.status_label.setText("开始训练...")
        
        # 启动训练
        self.worker.start()

    def update_env_view(self, frame):
        """更新环境视图"""
        if frame is None:
            return
            
        if isinstance(frame, np.ndarray):
            # 将numpy数组转换为QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 调整图像大小以适应标签
            scaled_pixmap = pixmap.scaled(
                self.env_view.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.env_view.setPixmap(scaled_pixmap)
        else:
            print(f"Unexpected frame type: {type(frame)}")

    def update_plot_view(self, plot_frame):
        """更新训练曲线视图"""
        if plot_frame is not None:
            # 转换为RGB格式
            plot_frame = cv2.cvtColor(plot_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = plot_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(plot_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 计算缩放比例，保持长宽比
            view_size = self.plot_view.size()
            scaled_pixmap = pixmap.scaled(
                view_size.width() * 0.95,  # 留出一些边距
                view_size.height() * 0.95,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.plot_view.setPixmap(scaled_pixmap)
            
    def update_progress(self, episode, total_episodes):
        """更新进度条和状态"""
        progress = int((episode / total_episodes) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"训练中... Episode {episode}/{total_episodes}")


class MainWindow(QMainWindow):
    """主窗口，管理场景选择和主操作界面"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("强化学习环境配置")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        self.main_operation = None  # 主操作界面实例

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 初始化场景选择窗口
        self.scene_selection = SceneSelectionWindow(self)  # 场景选择—— 界面1
        self.scene_selection.scene_selected.connect(self.show_main_window)
        self.layout.addWidget(self.scene_selection)

        # 应用 QSS 样式
        self.setStyleSheet(QSS_STYLE)

    def show_main_window(self, scene):
        """切换到主操作界面"""
        self.layout.removeWidget(self.scene_selection)
        self.scene_selection.deleteLater()
        self.main_operation = MainOperationWindow(scene, self)  # 创建第二个界面
        self.layout.addWidget(self.main_operation)

    def closeEvent(self, event):
        if self.main_operation and self.main_operation.worker:
            if self.main_operation.worker.isRunning():
                self.main_operation.worker.terminate()  # 停止工作线程
        return super().closeEvent(event)

    def __del__(self):
        try:
            if hasattr(self, 'maze_view'):
                self.maze_view.quit_game()
        except Exception as e:
            print(f"Error cleaning up maze environment: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    #from utils.pipe_frame import pipe_frame
    #pipe_frame.set_frame(frame)  # 更新管道中的 frame



