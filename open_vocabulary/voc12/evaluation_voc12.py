import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import scipy.io as sio
import argparse

from datasets import load_img_name_list_100
import sys
sys.path.insert(0, sys.path[0]+"/../..")

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

embs = [
        "plane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "table",
        "dog",
        "horse",
        "motorbike",
        "people",
        "plant",
        "sheep",
        "sofa",
        "train",
        "monitor"
    ]
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    # count=0
    txt = f'{predict_folder}/text.txt'
    
    f=open(txt,"a+")
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))
    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.mat'%name)
                # predict_file = os.path.join(predict_folder,'%s.mat'%str(idx))
                # if not os.path.exists(predict_file):
                #     print(predict_file)
                #     continue
                predict_dict=sio.loadmat(predict_file)
                
                try:
                    h, w = list(predict_dict.values())[-1].shape
                except:
                    # count+=1
                    print(name)
                    continue
                tensor = np.zeros((num_cls,h,w),np.float32)
                for key in list(predict_dict.keys())[3:]:
                    # index=embs.index(key)
                    tensor[int(key)+1] = predict_dict[key]/255
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            if predict.shape != gt.shape:
                continue
            mask = (predict==gt) * cal
            
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100
               
    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            if i%2 != 1:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(categories[i],IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()

class Args:
    comment='train1464'
    curve=True
    end=75
    base_dir = 'output/voc12_wo_sy_norm'
    image_dir = os.path.join(base_dir,'images')
    cam_npy_dir = os.path.join(base_dir,'images')
    gt_dir='/data/wjl/data/VOCdevkit/VOC2012/SegmentationClassAug'
    list='dataset/voc12/val_id.txt'
    # list='/root/autodl-tmp/wjl/ptp_diffusion/voc12/train_aug_id.txt'
    logfile=os.path.join(base_dir,'eval.txt')
    num_classes=21
    start=30
    t=None
    type='npy'           

import torch
import torch.nn.functional as F
def test():
    args = Args()

    categories = ['plane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'table','dog','horse','motorbike','people','plant','sheep','sofa','train','monitor']

    if not os.path.exists(args.cam_npy_dir):
        os.makedirs(args.cam_npy_dir)

    pred_name_list = os.listdir(args.image_dir)
    img_gt_name_list = open(args.list).readlines()
    img_name_list = [img_gt_name.strip() for img_gt_name in img_gt_name_list]
    tmp_list = []
    big_data_len = 10582
    shot_data_len = 1449

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True)
        writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        for i in range(args.start, args.end):
            t = i/100.0
            loglist = do_python_eval(args.cam_npy_dir, args.gt_dir, name_list, args.num_classes, args.type, t)
            l.append(loglist['mIoU'])
            print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            # else:
            #     break
        print('Best background score: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        writelog(args.logfile, {'mIoU':l, 'Best mIoU': max_mIoU, 'Best threshold': best_thr}, args.comment)

if __name__ == '__main__':
    test()