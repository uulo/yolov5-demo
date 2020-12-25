import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path
from utils.loss import compute_loss
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=True,
         log_imgs=0):  # number of logged images
    """
    param data: 数据集配置文件，数据集路径，类名等
    param weights: 测试的模型权重文件
    param batch_size: 前向传播时的批次, 默认32
    param imgsz: 输入图片分辨率大小, 默认640
    param conf_thres: 筛选框的时候的置信度阈值, 默认0.001
    param iou_thres: 进行NMS(非极大值抑制)的时候的IOU阈值, 默认0.6iou
    param save_json: 是否按照coco的json格式保存预测框，并且使用cocoapi做评估(需要同样coco的json格式的标签), 默认False
    param single_cls: 数据集是否只有一个类别，默认False
    param augment: 默认False
    param verbose: 是否打印出每个类别的mAP, 默认False
    param model: 测试的模型，训练时调用test传入
    param dataloader: 测试集的dataloader，训练时调用test传入
    param save_dir: 保存在测试时第一个batch的图片上画出标签框和预测框的图片路径
    param save_txt: 是否以txt文件的形式保存模型预测的框坐标, 默认False
    param save_hybrid: 默认False
    param save_conf: 默认False
    param plots:
    param log_imgs:
    """
    # Initialize/load model and set device
    # 判断是否在训练时调用test，如果是则获取训练时的设备
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        # 选择设备
        device = select_device(opt.device, batch_size=batch_size)

        # Directories
        # 目录
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # 加载模型
        model = attempt_load(weights, map_location=device)  # load FP32 model
        # 检查输入图片分辨率是否能被32整除
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    # 如果设备不是cpu并且gpu数目为1，则将模型由Float32转为Float16，提高前向传播的速度
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()  # to FP16

    # Configure
    model.eval()
    # 加载数据配置信息
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    # 设置iou阈值，从0.5~0.95，每间隔0.05取一次
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # iou个数
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        # 创建一个全0数组测试一下前向传播是否正常运行
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # 获取图片路径
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        # 创建dataloader
        # ？？？注意这里rect参数为True，yolov5的测试评估是基于矩形推理的
        dataloader = create_dataloader(path, imgsz, batch_size, model.stride.max(), opt, pad=0.5, rect=True)[0]

    # 初始化测试的图片数量
    seen = 0
    # 建立混淆矩阵
    confusion_matrix = ConfusionMatrix(nc=nc)
    # 获取类别的名字
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    """
    获取coco数据集的类别索引
    这里要说明一下，coco数据集有80个类别(索引范围应该为0~79)，
    但是他的索引却属于0~90(笔者是通过查看coco数据测试集的json文件发现的，具体原因不知)
    coco80_to_coco91_class()就是为了与上述索引对应起来，返回一个范围在0~90的索引数组
    """
    coco91class = coco80_to_coco91_class()
    # 设置tqdm进度条的显示信息
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # 初始化指标，时间
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    # 初始化测试集的损失
    loss = torch.zeros(3, device=device)
    # 初始化json文件的字典，统计信息，ap
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        img = img.to(device, non_blocking=True)
        # 图片也由Float32->Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width

        with torch.no_grad():
            # Run model # 开始测试
            """
            time_synchronized()函数里面进行了torch.cuda.synchronize()，再返回的time.time()
            torch.cuda.synchronize()等待gpu上完成所有的工作
            总的来说就是这样测试时间会更准确 
            """
            t = time_synchronized()
            # 前向传播
            # inf_out为预测结果, train_out训练结果
            inf_out, train_out = model(img, augment=augment)  # inference and training outputs
            # t0累计前向传播的时间
            t0 += time_synchronized() - t

            # Compute loss
            # 如果是在训练时进行的test，则通过训练结果计算并返回测试集的GIoU, obj, cls损失
            if training:
                loss += compute_loss([x.float() for x in train_out], targets, model)[1][:3]  # box, obj, cls

            # Run NMS
            # t1累计后处理NMS的时间
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            """
            non_max_suppression进行非极大值抑制;
            conf_thres为置信度阈值，iou_thres为iou阈值
            ？？？merge为是否合并框
            """
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb)
            t1 += time_synchronized() - t

        # Statistics per image
        # 为每一张图片做统计, 写入预测信息到txt文件, 生成json文件字典, 统计tp等
        for si, pred in enumerate(output):
            # 获取第si张图片的标签信息, 包括class,x,y,w,h
            # targets[:, 0]为标签属于哪一张图片的编号
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            # 获取标签类别
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            # 统计测试图片数量
            seen += 1

            # 如果预测为空，则添加空的信息到stats里
            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Append to text file
            # 保存预测结果为txt文件
            if save_txt:
                # 获得对应图片的长和宽
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    # xyxy格式->xywh, 并对坐标进行归一化处理
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # ？？？标签格式
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 保存预测类别和坐标到txt文件
                    with open(save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # W&B logging  # wandb日志
            if plots and len(wandb_images) < log_imgs:
                box_data = [{"position": {"minX": xyxy[0], "minY": xyxy[1], "maxX": xyxy[2], "maxY": xyxy[3]},
                             "class_id": int(cls),
                             "box_caption": "%s %.3f" % (names[cls], conf),
                             "scores": {"class_score": conf},
                             "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                wandb_images.append(wandb.Image(img[si], boxes=boxes, caption=path.name))

            # Append to pycocotools JSON dictionary
            # ？？若采用coco的json格式，save_json=True，则添加pycocotools的json字典
            # 采用之前保存的json格式预测结果，通过cocoapi评估指标
            # 需要注意的是 测试集的标签也需要转成coco的json格式
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                # 获取图片id
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                # detected用来存放已检测到的目标
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                # 获得xyxy格式的框并乘以wh
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                # 对图片中的每个类单独处理
                for cls in torch.unique(tcls_tensor):
                    # 标签框该类别的索引
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # 预测框该类别的索引
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # box_iou计算预测框与标签框的iou值，max(1)选出最大的ious值,i为对应的索引
                        """
                        pred shape[N, 4]
                        tbox shape[M, 4]
                        box_iou shape[N, M]
                        ious shape[N, 1]
                        i shape[N, 1], i里的值属于0~M
                        """
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            # 获得检测到的目标
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                # 添加d到detected
                                detected_set.add(d.item())
                                detected.append(d)
                                # iouv为以0.05为步长 0.5到0.95的序列
                                # 获得不同iou阈值下的true positive
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            # 每张图片的结果统计到stats里
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        # 画出前3个batch的图片的ground truth和预测框并保存
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(output), paths, f, names), daemon=True).start()

    # Compute statistics
    # 将stats列表的信息拼接到一起
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        # 根据上面得到的tp等信息计算指标
        # 精准度TP/TP+FP，召回率TP/P，map，f1分数，类别
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # ？？？nt是一个列表，测试集每个类别有多少个标签框
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    # 打印指标结果
    pf = '%20s' + '%12.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    # 细节展示每一个类别的指标
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    # 打印前向传播耗费的时间、nms的时间、总时间
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
    if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb and wandb.run:
            wandb.log({"Images": wandb_images})
            wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        # 获取预测框的json文件路径并打开
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            # 获取并初始化测试集标签的json文件
            anno = COCO(anno_json)  # init annotations api
            # 初始化预测框的文件
            pred = anno.loadRes(pred_json)  # init predictions api
            # 创建评估器
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            # 评估
            eval.evaluate()
            eval.accumulate()
            # 展示结果
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    # 返回测试指标结果
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    model.float()  # for training
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')    # 测试的模型权重文件
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='*.data path')     # 数据集配置文件，数据集路径，类名等
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')   # 前向传播时的批次, 默认32
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')     # 输入图片分辨率大小, 默认640
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')    # 筛选框的时候的置信度阈值, 默认0.001
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')     # 进行NMS(非极大值抑制)的时候的IOU阈值, 默认0.6iou
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")                  # 设置测试形式, 默认val, 具体可看下面代码解析注释
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')    # 测试的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')  # 数据集是否只有一个类别，默认False
    # 可将准确率提高若干个百分点，它就是测试时增强（test time augmentation, TTA）。这里会为原始图像造出多个不同版本，包括不同区域裁剪和更改缩放程度等，并将它们输入到模型中；然后对多个版本进行计算得到平均输出，作为图像的最终输出分数。
    # 这种技术很有效，因为原始图像显示的区域可能会缺少一些重要特征，在模型中输入图像的多个版本并取平均值，能解决上述问题。
    parser.add_argument('--augment', action='store_true', help='augmented inference')            # 测试时是否使用TTA(test time augmentation), 默认False
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')            # 是否打印出每个类别的mAP, 默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')         # 是否以txt文件的形式保存模型预测的框坐标, 默认False
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')    # 默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')     # 默认False
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')   # 是否按照coco的json格式保存预测框，并且使用cocoapi做评估(需要同样coco的json格式的标签), 默认False
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    opt = parser.parse_args()
    # 设置参数save_json
    opt.save_json |= opt.data.endswith('coco.yaml')
    # check_file检查文件是否存在
    opt.data = check_file(opt.data)  # check file
    print(opt)

    # task = 'val', 'test', 'study'
    # task = ['val', 'test']时就正常测试验证集、测试集
    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             )

    # task == 'study'时，就评估yolov5系列和yolov3-spp各个模型在各个尺度下的指标并可视化
    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json,
                               plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(f, x)  # plot
