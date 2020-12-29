from __future__ import print_function

import os
import sys
import xml.dom.minidom as xmldom


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
# from core.models.model_zoo import get_segmentation_model
import numpy as np
import argparse
import glob
import time
from PIL import Image

def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    print(eles.tagName)
    img_name = eles.getElementsByTagName("filename")[0].firstChild.data
    label_name = eles.getElementsByTagName("resultfile")[0].firstChild.data
    # ymin = eles.getElementsByTagName("ymin")[0].firstChild.data
    # ymax = eles.getElementsByTagName("ymax")[0].firstChild.data
    print(img_name, label_name)
    return img_name, label_name

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--in_dic', type=str, default=None)
    parser.add_argument('--out_dic', type=str, default=None)
    parser.add_argument('--numclass', type=int, default=7)
    args = parser.parse_args()
    return args

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        path = os.listdir(input_path)
        file_extension = '.' + path[0].split('.')[1]
        self.files = glob.glob(input_path + "/*" + file_extension)
        self.files.sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        BatchNorm2d = nn.BatchNorm2d
        # self.model = get_segmentation_model(model='DCnet_A1', dataset='isprs', backbone='resnet34',
        #                                     aux=False, pretrained=True, pretrained_base=False,
        #                                     local_rank=args.local_rank,classnum=numclass,
        #                                     norm_layer=BatchNorm2d, root='./trainedmodels/').to(self.device)
        # self.model.to(self.device)
        self.numclass = numclass
    def decode_seg_map_sequence(self, label_masks):
        rgb_masks = []
        label_colours = np.array([
         [0, 0, 0],
         [0, 255, 255],
         [255, 0, 0],
         [0, 0, 255],
         [0, 255, 0],
         [255, 255, 0],
         [255, 255, 255]])
        for label_mask in label_masks:
            r = label_mask.copy()
            g = label_mask.copy()
            b = label_mask.copy()
            for ll in range(0, numclass):
                r[label_mask == ll] = label_colours[ll, 0]
                g[label_mask == ll] = label_colours[ll, 1]
                b[label_mask == ll] = label_colours[ll, 2]
            rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
            rgb[:, :, 0] = r
            rgb[:, :, 1] = g
            rgb[:, :, 2] = b
            rgb_mask = rgb
            rgb_masks.append(rgb_mask)
        rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
        return rgb_masks

    def eval(self):
        with open('./result_s_pic.txt', "w") as f:
            f.close()
        with open('./result_a_pic.txt', "w") as f:
            f.close()
        # self.model.eval()
        # model = self.model
        total_time = 0 
        for file in self.files:
            #----------------------------- 图像读取
            start = time.process_time()
            # tree=ET.parse(file)
            # root = tree.getroot()
            # for obj in root.iter('object'):
            #     #difficult = obj.find('difficult').text
            #     #cls = obj.find('name').text
            #     #if cls not in classes or int(difficult)==1:
            #     #    continue
            #     name = obj.find('name').text
            #     print(name)
            [img_name,label_name] = parse_xml(file)
            # name = file.split('/')[-1].split('.')[0]
            img_file = ('\\').join(file.split('\\')[:-1]) + '\\' + img_name
            img = Image.open(img_file)
            img_VH_np = np.array(img)
            img_VH_np_high8 = img_VH_np >> 8
            img_VH_np_high8 = img_VH_np_high8.astype(np.uint8)
            img_VH_np_low8 = img_VH_np << 8
            img_VH_np_low8 = img_VH_np_low8 >> 8
            img_VH_np_low8 = img_VH_np_low8.astype(np.uint8)
            img = np.concatenate((img_VH_np_low8[np.newaxis, :], img_VH_np_high8[np.newaxis, :], img_VH_np_low8[np.newaxis, :]), axis=0)
            img = img.transpose(1, 2, 0)
            image = self.transform(img).unsqueeze(0)
            image = image.to(self.device)

            # outputs = model(image)
            # out_pred = self.decode_seg_map_sequence(torch.max((outputs[0])[:3], 1)[1].detach().cpu().numpy())[0].float()
        #     pred = out_pred.cpu().data.numpy()
        #     if os.path.isdir(save_path) is False:
        #         os.makedirs(save_path)
        #     pred_out = Image.fromarray(pred.transpose(1, 2, 0).astype('uint8')).convert('RGB')
        #     pred_out.save(save_path + name +'.png')
        #     end = time.process_time()
        #     s_pic_time = end-start
        #     total_time += s_pic_time
        #     if os.path.isfile('./result_s_pic.txt') is False:
        #         with open('./result_s_pic.txt', "w") as f:
        #             f.write(name + ': ' + str(s_pic_time) + 's' + "\n")
        #             f.close()
        #     else:
        #         with open('./result_s_pic.txt', "a") as f:
        #             f.write(name + ': ' + str(s_pic_time) + 's' + "\n")
        #             f.close()
        # average_time = total_time/len(self.files)
        # if os.path.isfile('./result_a_pic.txt') is False:
        #     with open('./result_a_pic.txt', "w") as f:
        #         f.write("************  The part of segmentation  ************" + "\n")
        #         f.write('Number of Pictures' + ': ' + str(len(self.files)) + "\n")
        #         f.write('Total Time' + ': ' + str(total_time) + 's' + "\n")
        #         f.write('Average Time' + ': ' + str(average_time) + 's' + "\n")
        #         f.close()
        # else:
        #     with open('./result_a_pic.txt', "a") as f:
        #         f.write("************  The part of segmentation  ************" + "\n")
        #         f.write('Number of Pictures' + ': ' + str(len(self.files)) + "\n")
        #         f.write('Total Time' + ': ' + str(total_time) + 's' + "\n")
        #         f.write('Average Time' + ': ' + str(average_time) + 's' + "\n")
        #         f.close()
        return len(self.files)


if __name__ == '__main__':
    start_all = time.process_time()
    args = parse_args()
    args.in_dic = os.path.join(os.getcwd(),'sar-data')
    args.out_dic = os.path.join(os.getcwd(),'sar-result')
    # args.in_dic = os.getcwd()
    input_path = args.in_dic
    save_path = args.out_dic
    numclass = args.numclass
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    evaluator = Evaluator(args)
    pic_num = evaluator.eval()
    torch.cuda.empty_cache()
    end_all = time.process_time()
    total_time_all = end_all - start_all
    average_time_all = total_time_all/pic_num
    if os.path.isfile('./result_a_pic.txt') is False:
        with open('./result_a_pic.txt', "w") as f:
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("************  Global  ************" + "\n")
            f.write('Number of Pictures' + ': ' + str(pic_num) + "\n")
            f.write('Total Time' + ': ' + str(total_time_all) + 's' + "\n")
            f.write('Average Time' + ': ' + str(average_time_all) + 's' + "\n")
            f.close()
    else:
        with open('./result_a_pic.txt', "a") as f:
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("************  Global  ************" + "\n")
            f.write('Number of Pictures' + ': ' + str(pic_num) + "\n")
            f.write('Total Time' + ': ' + str(total_time_all) + 's' + "\n")
            f.write('Average Time' + ': ' + str(average_time_all) + 's' + "\n")
            f.close()