# train.py

需要在网上下载预训练模型.pt到weights目录

考虑保存日志logging

可以更改model/yolov5s.yaml中的anchor。anchor使用了utils/dataset.py的kmeans自动anchor，默认自动更改

model yolov5s.yaml

***可以更改模型评价方法best_fitness # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得

单机多卡可使用DP(DataParallel模式)，单机多卡和多机多卡可使用DDP(DistributedDataParallel模式)

模型GPU进程数为[-1, 0]时，使用EMA指数滑动平均

参数resume预设false

models中yolov5s.yaml的nc不应该需要改

yolov5网络的总步长为32，所以其实只要图片边长能够整除32就可以，所以是否将img-size设为32的倍数即可

dataloader默认不使用矩形训练rect

testloader默认使用矩形训练rect