from __future__ import print_function, division
from torch.utils.data import DataLoader
import os
import scipy.io as sio
# from utils.load_data import  load_radar_data
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
import  torch
from torchvision import transforms
# from dataloaders import custom_transforms as tr




class SARData(Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 7

    def __init__(self,
                 args,
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        data_list = os.listdir(self._base_dir)
        self.img_list = []
        for _file in data_list:
            if _file.endswith(args.polar+'.tiff'):
                self.img_list.append(_file)
        self.img_list.sort(key=lambda x: int(x[:-8]))
        self.img_list = self.img_list[0:args.data_number]
        self.NUM_CLASSES = 7
        self.args = args
        print('Number of data： {:d}'.format(len(self.img_list)))

    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, index):
        img_path = os.path.join(self._base_dir,self.img_list[index])
        img = Image.open(img_path)
        img_VH_np = np.array(img)
        img_VH_np_high8 = img_VH_np >> 8
        img_VH_np_high8 = img_VH_np_high8.astype(np.uint8)
        img_VH_np_low8 = img_VH_np << 8
        img_VH_np_low8 = img_VH_np_low8 >> 8
        img_VH_np_low8 = img_VH_np_low8.astype(np.uint8)
        img = np.concatenate((img_VH_np_low8[np.newaxis, :], img_VH_np_high8[np.newaxis, :], img_VH_np_low8[np.newaxis, :]), axis=0)
        img = img.transpose(1, 2, 0)
        # image = self.transform(img).unsqueeze(0)
        image = self.transform(img)
        #
        label_path = img_path[:-8] + '_gt.png'
        label = Image.open(label_path)
        label = torch.from_numpy( np.array(label))
        sample = {'image': image, 'label': label}
        return sample
        # for split in self.split:
        #     if split == "train":
        #         return self.transform_tr(sample)
        #     elif split == 'val':
        #         return self.transform_val(sample)

    def transform(self,sample):
        composed_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([.485, .456, .406], [.229, .224, .225]),])
        return composed_transforms(sample)


 



# if __name__ == '__main__':
#     from dataloaders.utils import decode_segmap
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
#     import argparse

#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.base_size = 513
#     args.crop_size = 513

#     voc_train = VOCSegmentation(args, split='train')

#     dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

#     for ii, sample in enumerate(dataloader):
#         for jj in range(sample["image"].size()[0]):
#             img = sample['image'].numpy()
#             gt = sample['label'].numpy()
#             tmp = np.array(gt[jj]).astype(np.uint8)
#             segmap = decode_segmap(tmp, dataset='pascal')
#             img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
#             img_tmp *= (0.229, 0.224, 0.225)
#             img_tmp += (0.485, 0.456, 0.406)
#             img_tmp *= 255.0
#             img_tmp = img_tmp.astype(np.uint8)
#             plt.figure()
#             plt.title('display')
#             plt.subplot(211)
#             plt.imshow(img_tmp)
#             plt.subplot(212)
#             plt.imshow(segmap)

#         if ii == 1:
#             break

#     plt.show(block=True)
