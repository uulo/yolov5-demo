import os

# 训练，采用图像预载入内存、矩形训练、图像加权（防止各类别标签不均匀）
os.system('python train.py --cache-images --rect --image-weights')
os.system('detect.py')