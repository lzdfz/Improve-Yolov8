from ultralytics import YOLO
# 加载训练好的模型或者网络结构配置文件

model = YOLO(
'C:\\MQS\\Yolov8_10_12\\ultralytics-main\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8+CA.yaml')

# 打印模型参数信息
print(model.info(detailed=True))
print()
print()
print()
print()
print('      layers:层数      parameters：参数量      gradients：梯度数量       n GFLOPs”：即每秒 10*n 亿次浮点运算')

print(model.info())