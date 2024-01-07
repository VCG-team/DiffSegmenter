from typing import Optional, Union, Tuple, List, Callable, Dict
import sys
# sys.path.append("./../..")
sys.path.insert(0, sys.path[0]+"/../..")
import scipy.io as sio
import torch
from diffusers import StableDiffusionPipeline,  DDIMScheduler
import numpy as np
import abc
import ptp_utils
import visual_code.seq_aligner as seq_aligner
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset.mydatasets import build_dataset
import argparse
import os
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class LocalBlend:
    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer
from PIL import Image

def aggregate_all_attention(prompts,attention_store: AttentionStore, from_where: List[str], is_cross: bool, select: int):
    attention_maps = attention_store.get_average_attention()
    att_8 = []
    att_16 = []
    att_32 = []
    att_64 = []
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == 8*8:
                cross_maps = item.reshape(len(prompts), -1, 8, 8, item.shape[-1])[select]
                att_8.append(cross_maps)
            if item.shape[1] == 16*16:
                cross_maps = item.reshape(len(prompts), -1, 16, 16, item.shape[-1])[select]
                att_16.append(cross_maps)
            if item.shape[1] == 32*32:
                cross_maps = item.reshape(len(prompts), -1, 32, 32, item.shape[-1])[select]
                att_32.append(cross_maps)
            if item.shape[1] == 64*64:
                cross_maps = item.reshape(len(prompts), -1, 64, 64, item.shape[-1])[select]
                att_64.append(cross_maps)
    atts = []
    for att in [att_8,att_16,att_32,att_64]:
        att = torch.cat(att, dim=0)
        att = att.sum(0) / att.shape[0]
        atts.append(att.cpu())
    return atts

def aggregate_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out).mean(0)
    return out.cpu()


def show_cross_attention(prompts,attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0): 
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    texts = []
    maps = []
    tokens = tokenizer.encode(prompts[select])
    map = []
    for j in range(len(tokens)):
        map.append(attention_maps[:,:,j])

        texts.append(decoder(int(tokens[j])))
    maps.append(torch.stack(map))
    return images,texts,maps
    # ptp_utils.view_images(np.stack(images, axis=0))
    

def show_self_attention_comp(prompts, attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).view(res ** 2, res ** 2)
    # u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    # for i in range(max_com):
    #     image = vh[i].reshape(res, res)
    #     image = image - image.min()
    #     image = 255 * image / image.max()
    #     image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
    #     image = Image.fromarray(image).resize((256, 256))
    #     image = np.array(image)
    #     images.append(image)
    return images, attention_maps
    # ptp_utils.view_images(np.concatenate(images, axis=1))

def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None,t=100,noise_sample_num=2):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=t, guidance_scale=GUIDANCE_SCALE, generator=generator, low_resource=LOW_RESOURCE,noise_sample_num=noise_sample_num)
    # ptp_utils.view_images(images)
    # import ipdb;ipdb.set_trace()
    return images, x_t

def encode_imgs(imgs,vae):
    # imgs: [B, 3, H, W]
    imgs = 2 * imgs - 1
    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample() * 0.18215

    return latents

MY_TOKEN = None
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=1, type=int)

    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    # Dataset parameters
    parser.add_argument('--data-path', default='', type=str, help='dataset path')
    parser.add_argument('--img-list', default='', type=str, help='image list path')
    parser.add_argument('--data-set', default='', type=str, help='dataset')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    return parser

embs = [
        "plane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "buses",
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

# srcfile 需要复制、移动的文件   
# dstpath 目的地址
 
import os
import shutil
from glob import glob
 
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, f'{dstpath}/{fname}')          # 复制文件
        print ("copy %s -> %s"%(srcfile, f'{dstpath}/{fname}'))
                       
def read_json(json_path, choice):
    with open(json_path,"r") as tmp:
        data=json.load(tmp)
    re=data[choice[0]]
    if len(choice)==1:
        return re
    for key in re.keys():
        for i in range(1,len(choice)):
            re[key].extend(data[choice[i]][key])
        re[key]=list(set(re[key]))
    return re 
if __name__ == '__main__':
    output_path=f'./output/voc12_wo_sy_norm'
    img_output_path = f"{output_path}/images"
    attmap_output_path = f"{output_path}/self_att_maps"
    # prediced image class label by clip and blip
    json_path='dataset/voc12/predict_syn_txt_0.8_img_0.97.json'
    for paths in [output_path,img_output_path]:
        if not os.path.exists(paths):
            os.makedirs(paths)
    mycopyfile(os.path.abspath(__file__), output_path)

    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args(args=[])
    args.img_list='dataset/voc12'
    # if you want to run gt label, cls_labels_val.npy file has img class label
    args.label_file_path=None
    # your file path
    args.data_path='VOCdevkit/VOC2012'
    
    args.gen_attention_maps=False
    args.data_set='VOC12seg'
    dataset_train, args.nb_classes = build_dataset(is_train=False, args=args, is_big_data=True)
    sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    cudnn.benchmark = True
    blip_device = "cuda:2"
    diffusion_device = "cuda:2"

    dtype = torch.float16
    device = torch.device(diffusion_device) if torch.cuda.is_available() else torch.device('cpu')
    device1 = torch.device(diffusion_device) if torch.cuda.is_available() else torch.device('cpu')
    model_key = "runwayml/stable-diffusion-v1-5"
    # model_key = "stabilityai/stable-diffusion-2-1-base"
    # ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_key, local_files_only=True,torch_dtype=dtype).to(device)
    tokenizer = ldm_stable.tokenizer
    ldm_stable.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
    vae = ldm_stable.vae.to(device1)
    controller = AttentionStore()
    g_cpu = torch.Generator(4307)
    times = [100]
    weight = [0.3,0.5,0.1,0.1]

    
    processor = BlipProcessor.from_pretrained("blip/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("blip/blip-image-captioning-large").to(blip_device)
    prompt_texts = []
    # pred_classes=read_json(json_path,choice=['text_similarity','image_similarity','synonym_matching'])
    pred_classes=read_json(json_path,choice=['image_similarity','text_similarity'])
    

    print(len(data_loader_train))
    with torch.no_grad():
        for k,data in enumerate(data_loader_train):
            # if k<41:
            #     continue
            img_temp = data[0].permute(0, 2, 3, 1)
            h=data[2].shape[1]
            w=data[2].shape[2]
           
            orig_images = torch.zeros_like(img_temp)
            orig_images[:, :, :, 0] = img_temp[:, :, :, 0] #* 0.229 + 0.485 R
            orig_images[:, :, :, 1] = img_temp[:, :, :, 1] #* 0.224 + 0.456 G
            orig_images[:, :, :, 2] = img_temp[:, :, :, 2] #* 0.225 + 0.406 B
            orig_images[orig_images>1]=1
            orig_images[orig_images<0]=0

            rgb_512 = F.interpolate(orig_images.permute(0,3,1,2), (512, 512), mode='bilinear', align_corners=False).to(device1)
            rgb_512 = rgb_512 if dtype == torch.float32 else rgb_512.half()
            input_latent = encode_imgs(rgb_512,vae).to(device)
            noise_sample_num = 1
            noise = torch.randn([1,4,64,64]).to(device)
            noise = noise if dtype == torch.float32 else noise.half()
            images = []
            # x,y=torch.where(data[1])
            pred_class_key=data[3][0][-15:]
            # import ipdb;ipdb.set_trace()
            try:
                y=pred_classes[pred_class_key]
            except:
                print(pred_class_key)
                continue
            for i in range(0,len(y)): 
                controller.reset()
                prompt = f"a photograph of {y[i]}"
                img_path = data[3][0]
                raw_image = Image.open(img_path).convert('RGB')
                inputs = processor(raw_image, prompt, return_tensors="pt").to(blip_device)
                out = model.generate(**inputs)
                out_prompt = processor.decode(out[0], skip_special_tokens=True)

                prompt = [prompt + "++" + out_prompt[len(prompt):]]
                prompt_texts.append(f"{k} {prompt[0]}")
                print(k,prompt)
                imgs = []
                for t in times:
                    controller.reset()
                    latents_noisy = ldm_stable.scheduler.add_noise(input_latent, noise, torch.tensor(t,device=device))
                    latents_noisy = latents_noisy if dtype == torch.float32 else latents_noisy.half()
                    _, _ = run_and_display(prompt, controller, latent=latents_noisy,run_baseline=False, generator=g_cpu,t=t,noise_sample_num=noise_sample_num)
                    out_atts = []
                    cross_attention_maps = aggregate_all_attention(prompt,controller, ("up", "mid", "down"), True, 0)
                    self_attention_maps = aggregate_all_attention(prompt,controller, ("up", "mid", "down"), False, 0)
                    for idx,res in enumerate([8,16,32,64]):
                        if len(prompt[0].split(" "))>4 and prompt[0].split(" ")[4].endswith("ing"):
                            cross_att = cross_attention_maps[idx][:,:,[4,5]].mean(2).view(res,res).float()
                        else:
                            cross_att = cross_attention_maps[idx][:,:,[4]].mean(2).view(res,res).float()
                        if res != 64:
                            cross_att = F.interpolate(cross_att.unsqueeze(0).unsqueeze(0), size=(64,64), mode='bilinear', align_corners=False).squeeze().squeeze()
                        cross_att = (cross_att-cross_att.min())/(cross_att.max()-cross_att.min())
                        out_atts.append(cross_att * weight[idx])

                    cross_att_map = torch.stack(out_atts).sum(0).view(64*64,1)
                    self_att = self_attention_maps[3].view(64*64,64*64).float()

                    
                    att_map = torch.matmul(self_att,cross_att_map).view(res,res)

                    # att_map = (att_map-att_map.min())/(att_map.max()-att_map.min())
                    # att_map = F.sigmoid(8 * (att_map-0.4))
                    # att_map = (att_map-att_map.min())/(att_map.max()-att_map.min())

                    imgs.append(att_map)


                images.append(torch.mean(torch.stack(imgs,dim=0),dim=0).unsqueeze(0).repeat(3,1,1))
            # import ipdb;ipdb.set_trace()
            images = torch.stack(images)
            images=F.interpolate(images, size=(h,w), mode='bilinear', align_corners=False)
            pixel_max = images.max()
            for i in range(images.shape[0]):
                images[i]=((images[i]-images[i].min())/(images[i].max()-images[i].min()))*255
            cam_dict={}
            
            for i in range(0,len(y)): 
                class_emb= y[i]
                if class_emb == "buses":
                    class_emb = "bus"
                cam_dict[str(embs.index(y[i]))]=images[i].permute(1,2,0).cpu().numpy()[:,:,0]
            sio.savemat(os.path.join(img_output_path, img_path[-15:-4] + '.mat'),cam_dict,do_compression=True)

    with open(os.path.join(output_path,"prompts.txt"), 'w') as f:     
        f.writelines("\n".join(prompt_texts))

