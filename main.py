import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from model import NoisyCircle_Detector
from matplotlib.colors import Normalize
import torch

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
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def find_circle(img):
    # Fill in this function
    
    # Load the model
    model = NoisyCircle_Detector()
    saved_state_dict = torch.load('model.pth')
    model.load_state_dict(saved_state_dict) 
    model.eval()
    model.cuda()
    
    # Preprocess the image
    norm = Normalize(vmin = img.min(), vmax=img.max())
    img = norm(img)
    img = np.asarray(255*img,dtype=np.uint8)
    img = np.stack((img,img,img),axis=0) 
    img = np.expand_dims(img,axis=0)
    img = torch.from_numpy(img).float().cuda()

    # Get circle params
    output_params = model(img)
    output_params = torch.squeeze(output_params)
    output_params = output_params.data.cpu().numpy()
    row = output_params[0]
    col = output_params[1]
    rad = output_params[2]

    return (row, col, rad)


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

if __name__ == '__main__':
    main()
