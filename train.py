# import argparse
# from ultralytics import YOLO
# # import warnings
# # warnings.filterwarnings("ignore", category=UserWarning)
# # 明确指定 YOLOv8 的权重文件
# model = YOLO('yolov8n.pt')  # 使用 YOLOv8 的权重
#
# # 定义命令行参数
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', default='C:/MQS/Yolov8_10_12/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/123.yaml', help='数据集路径')
# parser.add_argument('--imgsz', type=int, default=640, help='输入图片大小')
# parser.add_argument('--batch', type=int, default=16, help='批次大小')
# parser.add_argument('--workers', type=int, default=48, help='工作进程数')
# parser.add_argument('--patience', type=int, default=50, help='耐心值，用于提前停止训练')
# parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
# parser.add_argument('--resume', action='store_true', help='是否从上次中断处继续训练')
# parser.add_argument('--device', type=str, default='0', help='使用的设备，例如 "0" 或 "0,1"')
#
# if __name__ == "__main__":
#     # 解析命令行参数
#     args = parser.parse_args()
#
#     # 设置模型配置文件和数据配置文件
#     model_yaml = "C:/MQS/Yolov8_10_12/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/yolov8+head+CA+BiFNP.yaml"
#     data_yaml = args.data
#
#     # 使用命令行参数设置训练相关属性
#     results = model.train(
#         task='detect',  # 任务类型为检测
#         data=data_yaml,  # 数据集配置文件
#         imgsz=args.imgsz,  # 输入图片大小
#         epochs=args.epochs,  # 训练轮数
#         batch=args.batch,  # 批次大小
#         workers=args.workers,  # 工作进程数
#         patience=args.patience,  # 提前停止训练的耐心值
#         resume=args.resume,  # 是否从上次中断处继续训练
#         device=args.device,  # 使用的设备
#     )


import argparse
from ultralytics import YOLO

# 定义命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='C:/MQS/Yolov8_10_12/ultralytics-main/ultralytics-main/ultralytics/cfg/datasets/123.yaml', help='数据集路径')
parser.add_argument('--imgsz', type=int, default=640, help='输入图片大小')
parser.add_argument('--batch', type=int, default=16, help='批次大小')
parser.add_argument('--workers', type=int, default=48, help='工作进程数')
parser.add_argument('--patience', type=int, default=50, help='耐心值，用于提前停止训练')
parser.add_argument('--epochs', type=int, default=150, help='训练轮数')
parser.add_argument('--resume', action='store_true', help='是否从上次中断处继续训练')
parser.add_argument('--device', type=str, default='0', help='使用的设备，例如 "0" 或 "0,1"')
parser.add_argument('--single_cls', type=bool, default=False, help='用于识别多种标签"')

if __name__ == "__main__":
    # 解析命令行参数
    args = parser.parse_args()

    # 设置模型配置文件和数据配置文件
    model_yaml = ("C:/MQS/Yolov8_10_12/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/yolov8+EMA.yaml")
    data_yaml = args.data

    # 加载自定义模型
    model = YOLO(model_yaml)

    # 使用命令行参数设置训练相关属性
    results = model.train(
        task='detect',  # 任务类型为检测
        data=data_yaml,  # 数据集配置文件
        imgsz=args.imgsz,  # 输入图片大小
        epochs=args.epochs,  # 训练轮数
        batch=args.batch,  # 批次大小
        workers=args.workers,  # 工作进程数
        patience=args.patience,  # 提前停止训练的耐心值
        resume=args.resume,  # 是否从上次中断处继续训练
        device=args.device,  # 使用的设备
    )