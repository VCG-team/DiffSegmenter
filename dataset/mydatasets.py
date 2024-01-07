import os
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image


def load_img_name_list_100(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list=[]
    img_id=[]
    for img_gt_name in img_gt_name_list:
        tmp=img_gt_name.strip().split(' ')
        img_name_list.append(tmp[0])
        img_id.append(tmp[1])
    return img_name_list,img_id
def load_img_name_list(dataset_path):

    img_gt_name_list = open(dataset_path).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]

    return img_name_list

def load_image_label_list_from_npy(img_name_list, label_file_path=None):
    if label_file_path is None:
        return None
    cls_labels_dict = np.load(label_file_path, allow_pickle=True).item()
    # print(cls_labels_dict)
    label_list = []
    for id in img_name_list:
        if id not in cls_labels_dict.keys():
            img_name = id + '.jpg'
        else:
            img_name = id
        label_list.append(cls_labels_dict[img_name])
    return label_list

class COCOClsDataset(Dataset):
    def __init__(self, img_name_list_path, coco_root, label_file_path, train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val_5k"}_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        # self.img_name_list=sorted(self.img_name_list)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)
        self.coco_root = coco_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx]

        if self.train or self.gen_attn :
            img = PIL.Image.open(os.path.join(self.coco_root, 'train2014', name + '.jpg')).convert("RGB")
            img_path=os.path.join(self.coco_root, 'train2014', name + '.jpg')
            
        else:
            img = PIL.Image.open(os.path.join(self.coco_root, 'val2014', name + '.jpg')).convert("RGB")
            img_path=os.path.join(self.coco_root, 'val2014', name + '.jpg')
        
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)

        return img, label,img_path

    def __len__(self):
        return len(self.img_name_list)



####分割重写数据集
class VOC12Dataset_seg(Dataset):
    def __init__(self, img_name_list_path, label_file_path,voc12_root, train=True, transform=None, gen_attn=False,is_big_data=True):
        if not train :
            img_name_list_path = os.path.join(img_name_list_path, f'val_id.txt')
        elif train and is_big_data:
            img_name_list_path = os.path.join(img_name_list_path, f'train_aug_id.txt')
        else:
            img_name_list_path = os.path.join(img_name_list_path, f'train_id.txt')
        self.img_name_list = load_img_name_list(img_name_list_path)
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train_100" if train or gen_attn else "val"}_id.txt')
        #     self.img_name_list= load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list,label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        mask= PIL.Image.open(os.path.join(self.voc12_root, 'SegmentationClassAug', name + '.png'))
        path = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
            # tmp=[]
            # tmp.append(transforms.Resize([224,224]))
            # tmp_=transforms.Compose(tmp)
            mask=torch.tensor(np.array(mask))
        return img, label, mask, path

    def __len__(self):
        return len(self.img_name_list)

class VOC10Dataset_seg(Dataset):
    def __init__(self, img_name_list_path,label_file_path, voc12_root, train=True, transform=None, gen_attn=False,is_big_data=True):
        if not train :
            img_name_list_path = os.path.join(img_name_list_path, f'val.txt')
        elif train and is_big_data:
            img_name_list_path = os.path.join(img_name_list_path, f'train_aug_id.txt')
        else:
            img_name_list_path = os.path.join(img_name_list_path, f'train.txt')
        self.img_name_list = load_img_name_list(img_name_list_path,)
        # else:
        #     img_name_list_path = os.path.join(img_name_list_path, f'{"train_100" if train or gen_attn else "val"}_id.txt')
        #     self.img_name_list= load_img_name_list(img_name_list_path)
        self.label_list = load_image_label_list_from_npy(self.img_name_list,label_file_path)
        self.voc12_root = voc12_root
        self.transform = transform

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        img = PIL.Image.open(os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')).convert("RGB")
        mask= PIL.Image.open(os.path.join(self.voc12_root, 'SegmentationClassContext', name + '.png'))
        path = os.path.join(self.voc12_root, 'JPEGImages', name + '.jpg')
        if not self.label_list:
            label=-1
        else:
            label = torch.from_numpy(self.label_list[idx])
        if self.transform:
            img = self.transform(img)
            # tmp=[]
            # tmp.append(transforms.Resize([224,224]))
            # tmp_=transforms.Compose(tmp)
            mask=torch.tensor(np.array(mask))
        return img, label, mask, path

    def __len__(self):
        return len(self.img_name_list)

def build_dataset(is_train, args, gen_attn=False, is_big_data=True):
    transform = build_transform(False, args)
    # transform = None
    dataset = None
    nb_classes = None

    if args.data_set == 'VOC12':
        dataset = VOC12Dataset(img_name_list_path=args.img_list, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 20
    elif args.data_set == 'VOC12seg':
        dataset = VOC12Dataset_seg(img_name_list_path=args.img_list, label_file_path=args.label_file_path,voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform,is_big_data=is_big_data)
        nb_classes = 20
    elif args.data_set == 'COCO':
        dataset = COCOClsDataset(img_name_list_path=args.img_list, coco_root=args.data_path, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'COCOMS':
        dataset = COCOClsDatasetMS(img_name_list_path=args.img_list, coco_root=args.data_path, scales=None, label_file_path=args.label_file_path,
                               train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 80
    elif args.data_set == 'VOC10seg':
        dataset = VOC10Dataset_seg(img_name_list_path=args.img_list, label_file_path=args.label_file_path, voc12_root=args.data_path,
                               train=is_train, gen_attn=gen_attn, transform=transform,is_big_data=is_big_data)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            # color_jitter=args.color_jitter,
            # auto_augment=args.aa,
            interpolation=args.train_interpolation,
            # re_prob=args.reprob,
            # re_mode=args.remode,
            # re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    # if resize_im and not args.gen_attention_maps:
    #     size = int((256 / 224) * args.input_size)
    #     t.append(
    #         transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    #     )
    #     t.append(transforms.CenterCrop(args.input_size))
    if False:
        t.append(transforms.Resize([args.input_size,args.input_size], interpolation=3))
    else:
        # t.append(transforms.Resize([args.input_size,args.input_size], interpolation=3))
        t.append(transforms.ToTensor())
        # t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
