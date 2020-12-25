import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    # 输入流目录，若是视频流可以用'rtsp://', 'rtmp://', 'http://'网址或txt文件输入批量视频流
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories  是否要新建目录
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize   初始化，如果使用cuda则使用FP16（16位浮点）进行半精度推断
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model   读取模型，检查图片大小
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16   使用半精度模型

    # Second-stage classifier  # 是否参用第二次分类器，默认False
    classify = False
    if classify:
        # initialize    ？？？若使用两步分类器，则使用resnet101
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader  # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        # cudnn的Benchmark模式会提升计算速度，但是由于计算中有随机性，每次网络前馈结果略有差异，？解释为根据卷积的大小不同，选择不同的快速卷积算法，类似快速傅里叶变换，用精度换时间
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # 如果检测视频的时候想显示出来，可以在这里加一行view_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors   # 设置名字和颜色
    names = model.module.names if hasattr(model, 'module') else model.names     # 得到模型中各类的名字
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]        # ？设置画框的颜色

    # Run inference  # 开始推断
    t0 = time.time()    # 推断开始时的时刻
    # 进行一次前向推理,测试程序是否正常
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img  初始化一个空图片
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once   只运行一次进行测试
    # 读取数据集的路径，图片，im0s？，vid_cap（帧数切片？）
    '''
    path 图片/视频路径
    img 进行resize+pad之后的图片
    img0 原size图片
    cap 当读取图片时为None，读取视频时为视频源
    '''
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32   将原始的uint8数据类型转换为fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0     # 将图片的rgb值的范围由0 - 255转换为0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()  # t1 = time.time()
        # print("preprocess_image:", t1 - t0)   # 图像预处理后的时刻，与t0间包含一次空白图片的运行测试

        # 进行前向传播
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]为预测框坐标
        预测框坐标为xywh(中心点+宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        """
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS   # 进行非极大值抑制
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()  # t2 = time.time()  推断后的时刻

        # Apply Classifier  # 是否参用第二次分类器，默认False
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections  # 推断后的数据处理
        for i, det in enumerate(pred):  # detections per image  # 对每一张图片作处理  ？？？det是一张图片的torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
            # ？？？如果输入源是webcam，则batch_size不为1，取出dataset中的一张图片
            if webcam:  # batch_size >= 1
                p, s, im0, frame = Path(path[i]), '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)

            # 设置保存图片/视频的路径
            save_path = str(save_dir / p.name)
            # 设置保存框坐标txt文件的路径
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            # 设置打印信息(图片长宽)
            s += '%gx%g ' % img.shape[2:]  # print string # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # ？？？normalization gain whwh 为由xyxy转化为xywh进行预处理
            if len(det):# 如果有推断数据
                # Rescale boxes from img_size to im0 size # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # ？？？此时坐标格式为xyxy
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results # 打印检测到的类别数量
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results # 保存预测结果
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # 在原图上画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS) # 打印前向传播+nms时间 1/时间等于帧数
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results # 如果设置展示，则show图片/视频
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections) # ？设置保存图片/视频
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    # 打印总时间
    print(f'Done. ({time.time() - t0:.3f}s)')  # time.time() 所有处理结束后的时刻
    # t0  推断开始时的时刻
    # t1  图像预处理后的时刻，与t0间包含一次空白图片的运行测试
    # t2  推断后的时刻
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')       # 设置推断所用权重文件保存在的目录
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam   #设置推断源，可以为目录或文件，0默认为网络摄像头
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')    # ？网络输入图片大小
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')    # 目标类别的置信度阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')           # 做nms目标IOU（交并比）的阈值
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')            # 设备选择（gpu或cpu） 0,1,2,3作用？
    parser.add_argument('--view-img', action='store_true', help='display results')                       # 是否要展示结果，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')                 # 是否将结果保存为txt(检测到的目标类别与包围框)，默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')# 是否保存目标类别的置信度到txt，默认False
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')  # 设置只保留某一部分类别，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')                # ？？？进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--augment', action='store_true', help='augmented inference')                    # ？？？推理的时候进行多尺度，翻转等操作(TTA)推理，默认False
    parser.add_argument('--update', action='store_true', help='update all models')                       # ？？？如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')         # 设置结果保存的项目的名字，默认False
    parser.add_argument('--name', default='exp', help='save results to project/name')                    # 设置结果保存的项目中文件夹的名字，默认False
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')# ？若存在项目与名字路径，默认False在原路径名加1新建路径，若选True则直接使用原路径，
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():  # 若选择更新模型，则进行更新
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
