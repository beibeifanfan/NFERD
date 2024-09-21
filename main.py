import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from datasets.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib

matplotlib.use('Agg')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default='./RD++_checkpoint_result', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--image_width', default=704 , type=int)  # 指定输入图像的尺寸，默认为256。
    parser.add_argument('--image_height', default=256, type=int)  # 指定输入图像的尺寸，默认为256。
    parser.add_argument('--detail_training', default='note', type=str)
    # parser.add_argument('--proj_lr', default=0.001, type=float)
    parser.add_argument( '--distill_lr', default=0.005, type=float)
    # parser.add_argument('--weight_proj', default=0.2, type=float)
    parser.add_argument('--classes', nargs="+", default=["fold3"])
    pars = parser.parse_args()
    return pars



def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_height, pars.image_width)

    train_path = 'ksdd_three_folds/' + _class_ + '/train/'
    test_path = 'ksdd_three_folds/' + _class_

    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path = pars.save_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'
    train_data = MVTecDataset_train(root=train_path, transform=data_transform)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print("train_samples: %d" % (len(train_dataloader) * pars.batch_size))
    print("test_samples: %d" % len(test_dataloader))

    # Use pretrained ImageNet for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)


    optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=pars.distill_lr,
                                         betas=(0.5, 0.999))

    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0

    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []


    total_loss = []

    history_infor = {}
    # set appropriate epochs for specific classes (Some classes converge faster than others)

    num_epoch=200

    print(f'with class {_class_}, Training with {num_epoch} Epoch')
    for epoch in tqdm(range(1, num_epoch + 1)):
        bn.train()
        decoder.train()
        total_loss_running = 0

        ## gradient acc
        accumulation_steps = 2

        for i, (img, _, _) in enumerate(train_dataloader):
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))

            L_distill = loss_fucntion(inputs, outputs)
            loss = L_distill

            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_distill.step()

                optimizer_distill.zero_grad()

            total_loss_running += loss.detach().cpu().item()

        auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, bn, decoder, test_dataloader, device)

        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)

        total_loss.append(total_loss_running / ((len(train_dataloader) * pars.batch_size)))

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(12, 8)
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        ax[0][0].plot(auroc_sp_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_px_list)
        ax[0][1].set_title('auroc_sp')
        ax[0][2].plot(aupro_px_list)
        ax[0][2].set_title('aupro_px')
        ax[1][2].plot(total_loss)
        ax[1][2].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_traning.png', dpi=100)

        print('Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(epoch, auroc_sp, auroc_px,
                                                                                             aupro_px))

        print('total_loss: {:.4f} ' .format( total_loss[epoch - 1]))


        epoch_metrics = {
            "epoch": epoch,
            "auroc_sp": auroc_sp,
            "auroc_px": auroc_px,
            "aupro_px": aupro_px,
            "total_loss": total_loss_running/ ((len(train_dataloader) * pars.batch_size))
        }

        with open(os.path.join(pars.save_folder, _class_, 'epoch_metrics.json'), 'a') as f:
            f.write(json.dumps(epoch_metrics))
            f.write('\n')

        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3

            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({
                        'decoder': decoder.state_dict(),
                        'bn': bn.state_dict()}, save_model_path)

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px


if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)
    all_classes = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
                   'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}

    # train all_classes
    # for c in all_classes:
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print(
            'Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp,
                                                                                                            auroc_px,
                                                                                                            aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)