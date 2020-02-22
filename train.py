import argparse
from torch.utils import data
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
from dataset import NoisyCircles
import eval_circle as eval_circle
from model import NoisyCircle_Detector
from torch import optim
import timeit
import numpy as np
import os.path as osp

BATCH_SIZE = 32
DATA_DIRECTORY = './'
DATA_LIST_PATH = './list.txt'
MAX_ITERS = 20000


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train circle detection under noise')
    
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS,
                        help="Number of training iterations.")

    return parser.parse_args()

def bbox_transform(deltas):
    pred_ctr_x = deltas[:, 0] 
    pred_ctr_y = deltas[:, 1] 
    pred_r = deltas[:, 2] 

    x1 = pred_ctr_x - pred_r
    y1 = pred_ctr_y - pred_r
    x2 = pred_ctr_x + pred_r
    y2 = pred_ctr_y + pred_r

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def giou_loss(output, target):
    """
    This function returns generalised iou loss for circle parameter estimation.
    Originally used for object detection (https://arxiv.org/pdf/1902.09630.pdf)
    """
    batch_size = output.size(0)

    x1, y1, x2, y2 = bbox_transform(output)
    x1g, y1g, x2g, y2g = bbox_transform(target)
    
    x2 = torch.max(x1, x2)
    y2 = torch.max(y1, y2)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    intsctk = torch.zeros(x1.size()).to(output)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    intsctk[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + 1e-7
    iouk = intsctk / unionk

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    miouk = iouk - ((area_c - unionk) / area_c)
    miouk = (1 - miouk).sum(0) / batch_size
 
    return miouk


def main():
    """Main function"""

    args = parse_args()
    print('Called with args:')
    print(args)
    
    model = NoisyCircle_Detector()
    model.cuda()
   
    params = []
    train_params = 0
    for _, value in model.named_parameters():
        if value.requires_grad:
            train_params += value.numel()
            params.append(value)
    print('Total no. of trainable parameters: %d' % train_params) 
    
    ### Optimizer 
    optimizer = optim.Adam(params, lr=0.001)
    optimizer.zero_grad()
    
    ### Dataloader
    trainloader = data.DataLoader(NoisyCircles(args.data_dir, args.data_list, max_iters=args.max_iters*args.batch_size), 
                    batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)


    model.train()
    loss_val = 0
    epoch = 0
    start = timeit.default_timer()
    f = open('output.txt','a')
    for i_iter, batch in enumerate(trainloader):
        images, params = batch
        images = Variable(images).cuda()
        optimizer.zero_grad()
        output = model(images)
        params = Variable(params).cuda()
        loss = giou_loss(output, params)
        loss.backward()
        optimizer.step()
        loss_val += loss.data.cpu()

        # Evaluate every 300 iterations 
        if i_iter % 300 == 0 and i_iter!=0:
            epoch += 1
            loss_val /= 300
            savedmodel_path = osp.join(args.data_dir, 'models', 'Epoch_%.2d.pth' % epoch)
            print('Saving model.....')
            torch.save(model.state_dict(),savedmodel_path)
            print('Evaluating....')
            val_loss, val_map = eval_circle.eval_circle(savedmodel_path)
            stats = 'Epoch: %d, Training Loss: %f, Validation Loss: %f, Validation mAP: %f' % (epoch,loss_val, val_loss, val_map)
            print(stats)
            f.write('%s\n' % stats) 
            loss_val = 0
    
    end = timeit.default_timer()
    print(end-start,'seconds') 
    f.write('Total time taken (including training and evaluation): %f seconds' % (end-start))
    f.close()

if __name__ == '__main__':
    main()
