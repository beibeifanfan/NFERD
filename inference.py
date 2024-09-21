from PIL import Image
import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj, cal_anomaly_map, cvt2heatmap, show_cam_on_image, \
    threshold_anomaly_map, min_max_norm
from utils.utils_train import ProjLayer
from utils.utils_train import MultiProjectionLayer
from datasets.dataset import MVTecDataset_test, get_data_transforms
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default='./RD++_checkpoint_result', type=str)
    parser.add_argument('--image_width', default=256, type=int)  # 指定输入图像的尺寸，默认为256。
    parser.add_argument('--image_height', default=256, type=int)  # 指定输入图像的尺寸，默认为256。
    parser.add_argument('--classes', nargs="+", default=["transistor"])
    pars = parser.parse_args()
    return pars

def inference(_class_, pars):
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_height, pars.image_width)

    test_path = 'mvtec/' + _class_

    checkpoint_class = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    print("test_samples: %d" % len(test_dataloader))

    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    checkpoint_class = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])

    auroc_px, auroc_sp, aupro_px  = evaluation_multi_proj(encoder,bn, decoder, test_dataloader, device)

    print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_, auroc_sp, auroc_px,
                                                                                     aupro_px))
    if not os.path.exists(pars.checkpoint_folder + '/' +  _class_ +'_heatmaps'):
        os.makedirs(pars.checkpoint_folder + '/' + _class_ +'_heatmaps')


    return auroc_sp, auroc_px, aupro_px

if __name__ == '__main__':
    pars = get_args()

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}

    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'{pars.checkpoint_folder}/metrics_checkpoints.csv', index=False)