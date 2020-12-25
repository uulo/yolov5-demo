# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy
from utils.torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

# 此函数根据图片的信息获取图片的宽、高信息
def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

# 根据LoadImagesAndLabels创建dataloader
def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False):
    """
    参数解析：
    path： 包含图片路径的txt文件或者包含图片的文件夹路径
    imgsz： 网络输入图片大小
    batch_size:批次大小
    stride：网络下采样最大总步长
    opt： 调用train.py时传入的参数，这里主要用到opt.single_cls，是否是单类数据集
    hyp： 网络训练时的一些超参数，包括学习率等，这里主要用到里面一些关于数据增强(旋转、平移等)的系数
    augment： 是否进行数据增强
    cache： 是否提前缓存图片到内存，以便加快训练速度
    pad： 设置矩形训练的shape时进行的填充
    rect： 是否进行矩形训练
    ？？？rank
    ？？？world_size
    ？？？workers
    ？？？image_weights
    """
    # 确保DDP中只有第一个进程首先处理数据集，而后面的其他进程可以使用缓存
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      rank=rank,
                                      image_weights=image_weights)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path  # os.path.abspath(p)返回p的绝对路径
        # 如果采用正则化表达式提取图片/视频，直接使用glob获取文件路径
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        # 如果path是一个文件夹，使用glob获取全部文件路径
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        # 是文件则直接获取
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        # os.path.splitext分离文件名和后缀(后缀包含.)
        # 分别提取图片和视频文件路径
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        # 图片与视频数量
        ni, nv = len(images), len(videos)

        self.img_size = img_size  # 输入图片size
        self.files = images + videos  # 整合图片和视频路径到一个列表
        self.nf = ni + nv  # number of files  # 总的文件数量
        # 设置判断是否为视频的bool变量，方便后面单独对视频进行处理
        self.video_flag = [False] * ni + [True] * nv
        # 初始化模块信息，代码中对于mode=images与mode=video有不同处理
        self.mode = 'image'
        # 如果包含视频文件，这初始化opencv中的视频模块，cap=cv2.VideoCapture等
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        # nf如果小于0，则打印提示信息
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        # self.count == self.nf表示数据读取完了
        if self.count == self.nf:
            raise StopIteration
        # 获取文件路径
        path = self.files[self.count]

        # 如果该文件为视频
        if self.video_flag[self.count]:
            # Read video  # 读取视频，修改mode为video
            self.mode = 'video'
            # 获取当前帧 画面，ret_val为一个bool变量，直到视频读取完毕之前都为True
            ret_val, img0 = self.cap.read()
            # 如果当前视频读取结束，则读取下一个视频
            if not ret_val:
                self.count += 1
                # 释放视频对象
                self.cap.release()
                # self.count == self.nf表示视频已经读取完了
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            # 当前读取的帧数
            self.frame += 1
            # 打印信息
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image  # 读取图片
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            # 打印信息
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        # 对图片进行resize+pad
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        # BGR转为RGB格式，channel轴换到前面
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # 将数组内存转为连续，提高运行速度，(不转的话也可能会报错)
        img = np.ascontiguousarray(img)
        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image

        # 返回路径，resize+pad的图片，原始图片，视频对象
        return path, img, img0, self.cap

    def new_video(self, path):
        # frame用来记录帧数
        self.frame = 0
        # 初始化视频对象
        self.cap = cv2.VideoCapture(path)
        # 视频文件中的总帧数
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference # 读取单个网络摄像头图像数据，detect.py代码中未使用
    def __init__(self, pipe='0', img_size=640):
        self.img_size = img_size

        if pipe.isnumeric():
            pipe = eval(pipe)  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        # 初始化mode为images
        self.mode = 'stream'
        self.img_size = img_size
        # 如果sources为一个保存了多个视频流的文件
        # 获取每一个视频流，保存为一个列表
        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        # 视频流个数
        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream # 开启线程读取视频流的帧
            # 打印当前视频/总视频数/视频流地址
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            # 如果source=数字则打开摄像头，否则打开视频流地址
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            # 获取视频的宽和长
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 获取视频的帧率
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            # 读取当前画面
            _, self.imgs[i] = cap.read()  # guarantee first frame
            # 创建多线程读取视频流，daemon表示主线程结束时子线程也结束
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        # 获取进行resize+pad之后的shape，letterbox函数默认(参数auto=True)是按照矩形推理进行填充
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            # 每4帧读取一次
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        # 对图片进行resize+pad
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        # 将读取的图片拼接到一起
        img = np.stack(img, 0)  # ？？？print(img.shape)，batch_size=1，多线程

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(img_paths):  # 由图片路径寻找标签路径
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


# 读取图片和标签
class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        self.img_size = img_size  # 输入图片分辨率大小
        self.augment = augment    # 数据增强
        self.hyp = hyp            # 超参数
        self.image_weights = image_weights     # 图片采样
        self.rect = False if image_weights else rect        # 矩形训练
        # mosaic数据增强
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        # mosaic增强的边界
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride         # 模型下采样的总步长

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                # 获取数据集路径path，包含图片路径的txt文件或者包含图片的文件夹路径
                # 使用pathlib.Path生成与操作系统无关的路径，因为不同操作系统路径的‘/’会有所不同
                p = Path(p)  # os-agnostic
                # 如果路径path为包含图片的文件夹路径
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # 如果路径path为包含图片路径的txt文件
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        # 获取图片路径，更换相对路径
                        t = t.read().strip().splitlines()
                        # 获取数据集路径的上级父目录，os.sep为路径里的破折号(不同系统路径破折号不同，os.sep根据系统自适应)
                        parent = str(p.parent) + os.sep
                        # ？？'/'替换为os.sep，os.path.splitext(x)将文件名与扩展名分开并返回一个列表
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise Exception('%s does not exist' % p)
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, 'No images found'
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels  # 获取数据集的标签
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        # 漏掉的标签数量，找到的标签数量，空的标签数量，重复的标签数量，总的标签数量
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=desc, total=n, initial=n)
        assert nf > 0 or not augment, f'No labels found in {cache_path}. Can not train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        # 数据集的数量
        n = len(shapes)  # number of images
        # 获取batch的索引
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        # 一个轮次batch的数量
        nb = bi[-1] + 1  # number of batches

        self.batch = bi  # batch index of image  # 图片批的总数
        self.n = n       # number of images      # 图像的总数量
        self.indices = range(n)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        # 矩形训练
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # 获取根据ar从小到大排序的索引
            irect = ar.argsort()
            # 根据索引排序数据集与标签路径、shape、h/w
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            # 初始化shapes，nb为一轮批次batch的数量
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                # 如果一个batch中最大的h/w小于1，则此batch的shape为(img_size*maxi, img_size)
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果一个batch中最小的h/w大于1，则此batch的shape为(img_size, img_size/mini)
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        # 初始化图片到内存，加速训练（注意：大型数据集可能导致内存溢出）
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path=Path('./labels.cache')):
        # Cache dataset labels, check images and read shapes
        # 初始化标签，检查图片及其形状
        x = {}  # dict
        # 初始化漏掉的标签数量，找到的标签数量，空的标签数量，重复的标签数量
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images  # 检查图片
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels  # 检查标签
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print('WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f"New cache created: {path}")
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        """
        self.indices在train.py中设置
        要配合着train.py中的以下代码配合使用
        image_weights为根据标签中每个类别的数量设置的图片采样权重
        如果image_weights=True，则根据图片采样权重获取新的下标
        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx
        """
        index = self.indices[index]  # linear, shuffled, or image_weights

        # 超参数
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            # 使用mosaic数据增强方式加载
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            # 加载图片并根据设定的输入大小与图片原大小的比例ratio进行resize(未做填充pad到正方形)
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # letterbox就是在保持纵横比的前提下对图像做resize，先resize然后按需要在周围pad上0像素
            # 如果进行矩形训练，则获取每个batch的输入图片的shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # 根据shape对图片做resize和pad填充，返回resize+pad之后的图片、缩放因子ratio、填充大小pad
            # 如果未进行矩形训练，则只进行pad填充到正方形
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels # 读取标签
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                # 根据pad调整框的标签坐标，并从归一化的xywh->未归一化的xyxy
                labels = x.copy()
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            # 图像增强
            if not mosaic:
                # 随机对图片进行旋转，平移，缩放，裁剪
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            # 随机改变图片的色调（H），饱和度（S），亮度（V）
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        nL = len(labels)  # number of labels
        if nL:

            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh # 调整框的标签，xyxy to xywh
            # 重新归一化标签0 - 1  # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:
            # 图片随机上下翻转
            # flip up-down
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # 图片随机左右翻转
            # flip left-right
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        # 初始化标签框对应的图片序号，配合下面的collate_fn使用
        labels_out = torch.zeros((nL, 6))
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes


    """
    pytorch的DataLoader打包一个batch的数据集时要经过此函数进行打包
    通过重写此函数实现标签与图片对应的划分，一个batch中哪些标签属于哪一张图片,形如
    [[0, 6, 0.5, 0.5, 0.26, 0.35],
     [0, 6, 0.5, 0.5, 0.26, 0.35],
     [1, 6, 0.5, 0.5, 0.26, 0.35],
     [2, 6, 0.5, 0.5, 0.26, 0.35],]
     前两行标签属于第一张图片，第三行属于第二张。。。
    """
    @staticmethod    # 返回函数的静态方法
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


# Ancillary functions --------------------------------------------------------------------------------------------------
# 以下为辅助函数
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        # 根据ratio选择不同的插值方式
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    # 分离通道
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    # 随机调整hsv之后重新组合通道
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    # 将hsv格式转为BGR格式
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    # loads images in a mosaic

    labels4 = []
    s = self.img_size
    # 随机取mosaic中心点
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    # 随机取其他三张图片的索引
    indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left  # top left左上角
            # 初始化大图
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 设置大图上的位置（左上角）
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 选取小图上的位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right  # top right右上角
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left  # bottom left左下角
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right   # bottom right右下角
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将小图上截取的部分贴到大图上
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算小图到大图上时所产生的偏移，用来计算mosaic增强后的标签框的位置
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        x = self.labels[index]
        labels = x.copy()
        # 重新调整标签框的位置
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        # 调整标签框在图片内部
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # 进行mosaic的时候将四张图片整合到一起之后shape为[2*img_size, 2*img_size]
    # 对mosaic整合的图片进行随机旋转、平移、缩放、裁剪，并resize为输入大小img_size
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # 计算缩放因子
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    """
    缩放(resize)到输入大小img_size的时候，如果没有设置上采样的话，则只进行下采样
    因为上采样图片会让图片模糊，对训练不友好影响性能。
    """
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 获取最小的矩形填充
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    # 如果scaleFill=True,则不进行填充，直接resize成img_size,任由图片进行拉伸和压缩
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 计算上下左右填充大小
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 进行填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# 随机透视变换 使用齐次坐标的变换矩阵（旧版为仿射变换）
# 仿射变换是透视变换的子集
# 仿射保持了二维图形的“平直性”（直线经过变换之后依然是直线）和“平行性”（二维图形之间的相对位置关系保持不变，平行线依然是平行线，且直线上点的位置顺序不变）。
def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center  # 平移到图片中心的平移矩阵系数
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective  # 设置随机透视矩阵系数
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale  # 设置随机旋转和缩放矩阵系数
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # 设置随机裁剪矩阵系数
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 设置随机平移矩阵系数
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 融合成透视矩阵并作用在图片上
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 调整框的标签
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        # 去除进行上面一系列操作后，被裁剪过小的框(面积小于裁剪前的20%)
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets

# 计算候选包围盒是否可用
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

# 应用图片cutout增强   https://arxiv.org/abs/1708.04552
def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels

# 创建目录
def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


# 通过将所有文件放在顶层来平铺递归目录
# Flatten a recursive directory by bringing all files to top level
def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


# 将检测数据集转换为分类数据集，每类一个目录
def extract_boxes(path='../coco128/'):  # from utils.datasets import *; extract_boxes('../coco128')
    # Convert detection dataset into classification dataset, with one directory per class

    path = Path(path)  # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


# autosplit方法可以自动将数据集切分成train、val、test
def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0)):  # from utils.datasets import *; autosplit('../coco128')

    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    # Arguments
        path:       Path to images directory
        weights:    Train, val, test weights (list)
    """
    path = Path(path)  # images dir
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split
    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]  # remove existing
    for i, img in tqdm(zip(indices, files), total=n):
        if img.suffix[1:] in img_formats:
            with open(path / txt[i], 'a') as f:
                f.write(str(img) + '\n')  # add image to txt file
