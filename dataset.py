import os.path as osp
import numpy as np
import cv2
from torch.utils import data

class NoisyCircles(data.Dataset):
    def __init__(self, root, list_path, max_iters=None):
        self.root = root
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "train_images/%s" % name)
            id_num = name.split('_')[1].split('.')[0]
            label_file = osp.join(self.root, "train_labels/ann_%s.txt" % id_num)
            self.files.append({
                "img": img_file,
                "label": label_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        with open(datafiles["label"]) as f:
            label = f.readline().split() 
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        image = image.transpose((2, 0, 1))  

        return image.copy(), label.copy()
