import random
import torch
import torch.utils.data as data
from utils import *
import scipy.io as sio

def arguement(x):
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)

    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=[1, 2])

    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, [1])

    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, [2])

    return x

class Dataset2(data.Dataset):

    def __init__(self, args, mask, mask_3d_shift, isTrain=False):
        self.isTrain = isTrain
        #self.data_set = data_set
        self.crop_size = args.crop_size
        self.mask = torch.from_numpy(mask)#.cuda()      # 256,256
        self.mask_3d = torch.from_numpy(mask_3d_shift).permute(2, 0, 1)#.cuda()     # 28,256,310
        self.noise_level = args.noise_level
        if isTrain:
            self.num = args.data_num
            self.data_path = args.data_path
            self.scene_list = os.listdir(args.data_path)
            self.scene_list.sort()
        else:
            self.num = 10
            self.data_path = args.test_path
            self.scene_list = os.listdir(args.test_path)
            self.scene_list.sort()

    def __getitem__(self, idx):
        if self.isTrain:
            index = random.randint(0, 204)
            scene_path = self.data_path + self.scene_list[index]
            img_dict = sio.loadmat(scene_path)
            if "img_expand" in img_dict:
                img = img_dict['img_expand'] / 65536.
            elif "img" in img_dict:
                img = img_dict['img'] / 65536.
            elif "data_slice" in img_dict:
                img = img_dict['data_slice'] / 65536.
            img = img.astype(np.float32)
        else:
            index = idx
            scene_path = self.data_path + self.scene_list[index]
            img_dict = sio.loadmat(scene_path)
            if 'img' in img_dict:
                img = img_dict['img']
            elif 'data_slice' in img_dict:
                img = img_dict['data_slice']

            img = img.astype(np.float32)
        #HSI = img#self.data_set[index]
        HSI = torch.from_numpy(img).permute(2, 0, 1)#.cuda()

        _, h, w = HSI.size()
        ix = random.randint(0, h - self.crop_size)
        iy = random.randint(0, w - self.crop_size)
        HSI = HSI[:, ix:ix+self.crop_size, iy:iy+self.crop_size]

        if self.isTrain:
            # 数据增强
            HSI = arguement(HSI)

        meas = shift_dataset(HSI + torch.randn_like(HSI)*self.noise_level/255,self.mask)     # 256,310
        Phi = self.mask_3d                  # 28,256,310
        PhiPhiT = torch.sum(Phi**2, dim=0)  # 256,310
        PhiPhiT[PhiPhiT == 0] = 1

        return meas, HSI, Phi, PhiPhiT

    def __len__(self):
        return self.num
