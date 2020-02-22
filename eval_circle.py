import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import cv2
import torch
from torch.autograd import Variable
from model import NoisyCircle_Detector
from matplotlib.colors import Normalize

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
    iouk = (1 - iouk).sum(0) / batch_size
    miouk = (1 - miouk).sum(0) / batch_size
 
    return miouk

def bbox_transform(deltas):
    pred_ctr_x = deltas[0] 
    pred_ctr_y = deltas[1] 
    pred_r = deltas[2] 

    x1 = pred_ctr_x - pred_r
    y1 = pred_ctr_y - pred_r
    x2 = pred_ctr_x + pred_r
    y2 = pred_ctr_y + pred_r

    return x1.view(-1), y1.view(-1), x2.view(-1), y2.view(-1)

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float32)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(model, img, params):
    output = model(img)
    output = torch.squeeze(output)
    params = np.array(params)
    params = torch.from_numpy(params).float()
    target = Variable(params).cuda()
    
    # Validation Loss
    val_loss = giou_loss(output, target)
    val_loss = val_loss.data.cpu().numpy()
    
    # Output params
    output = output.data.cpu().numpy()
    row = output[0]
    col = output[1]
    rad = output[2]

    return (row, col, rad), val_loss


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )

def preprocess(img):
    
    norm = Normalize(vmin = img.min(), vmax=img.max())
    img = norm(img)
    img = np.asarray(255*img,dtype=np.uint8)
    img = np.stack((img,img,img),axis=0)  
    img = np.expand_dims(img,axis=0)
    img = torch.from_numpy(img).float().cuda()

    return img

def eval_circle(path):
    results = []
    
    # Load model
    model = NoisyCircle_Detector()
    saved_state_dict = torch.load(path)
    model.load_state_dict(saved_state_dict) 
    model.eval()
    model.cuda()
    
    val_loss_total = 0
    for i in range(1000):
        params, img = noisy_circle(200, 50, 2)
        img = preprocess(img)
        detected, val_loss = find_circle(model, img, params)
        val_loss_total += val_loss
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())
    return val_loss_total/1000, (results > 0.7).mean()
