import torch,os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from base.utils import instantiate_from_config, get_device 

# 使用 HuggingFace 镜像站，避免直接访问 huggingface.co
import os as _os
_os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def load_eeg_data(config):
    exp_setting = config.get('exp_setting', 'intra-subject')
    
    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config,mode='test')
        print('init test_dataset success')
        train_dataset = EEGDataset(config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=25, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32, pin_memory=True)
        return train_loader, test_loader,test_loader
    
    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config,mode='test')
        print('init test_dataset success')
        
        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config,mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config,mode='train')
        print('init train_dataset success')
        test_loader = DataLoader(test_dataset, batch_size=config['data']['test_batch_size'], shuffle=False, drop_last=False,num_workers=25)#, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False,num_workers=32)#, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=32)#, pin_memory=True)
        return train_loader, val_loader, test_loader
    
class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config= config
        self.data_dir = config['data']['data_dir']
        # self.img_directory = os.path.join(self.data_dir,'../','Image_set_Resize',f'{mode}_images')
        # self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        # self.all_class_names.sort()
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels
    
        self.avg = config['data'][f"{mode}_avg"]

        self.blur_type = config['data']['blur_type']
        self.return_multi_blurs = bool(config['data'].get('return_multi_blurs', False))

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 4 if self.mode=='train' else 80

        # 中间层局部tokens配置
        self.token_layers = config['data'].get('token_layers', [6, 9, 12])
        self.max_tokens_per_image = int(config['data'].get('max_tokens_per_image', 64))

        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)

        data_dir = os.path.join(self.data_dir,'../Image_feature',f"{config['data']['blur_type']['target'].rsplit('.',1)[-1]}")
        os.makedirs(data_dir,exist_ok=True)

        features_filename = os.path.join(data_dir,f"{self.name}_{mode}.pt")

        pretrain_map= {
                'RN50':{'pretrained':'openai','resize':(224,224)}, #1024 
                'RN101':{'pretrained':'openai','resize':(224,224)}, #512
                'ViT-B-16':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #512
                'ViT-B-32':{'pretrained':'laion2b_s34b_b79k','resize':(224,224)}, #512
                'ViT-L-14':{'pretrained':'laion2b_s32b_b82k','resize':(224,224)}, #768
                'ViT-H-14':{'pretrained':'/hy-tmp/Brain2Visual_0923/premodals/CLIP-ViT-H-14/open_clip_pytorch_model.bin','resize':(224,224)}, #1024
                'ViT-g-14':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #1024
                'ViT-bigG-14':{'pretrained':'laion2b_s39b_b160k','resize':(224,224)}, #1280
            }

        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift,tag in zip([-self.c,0,self.c],['low','medium','high']):
                blur_param = config['data']['blur_type']
                blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size']+shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])
        # 为局部tokens选择使用的模糊等级：默认在启用不确定性时使用低模糊（清晰图像），否则使用中等
        self.token_blur_level = config['data'].get('token_blur_level', 'low' if self.config['data']['uncertainty_aware'] else 'medium')
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))] #transforms.Resize(pretrain_map[self.model_type]['resize']), 
        self.process_transform = transforms.Compose(process_term)

        self.match_label = np.ones(self.trial_all_subjects, dtype=int)

        if  os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=False)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
            self.img_local_tokens = saved_features.get('img_local_tokens', None)
            saved_tokens_blur = saved_features.get('tokens_blur_level', None)
            # 兼容：若此前以多级模糊保存，而当前不启用不确定性（单视图），
            # 则回退为 'medium' 或 'avg' 子字典，避免 __getitem__ 直接索引文件名时报错
            if not self.config['data']['uncertainty_aware'] and isinstance(self.img_features, dict):
                keys = set(self.img_features.keys())
                if ('medium' in keys) and isinstance(self.img_features['medium'], dict):
                    self.img_features = self.img_features['medium']
                elif ('avg' in keys) and isinstance(self.img_features['avg'], dict):
                    self.img_features = self.img_features['avg']
            # 若不存在局部tokens或模糊等级与期望不一致，则重新计算
            if (self.img_local_tokens is None) or (saved_tokens_blur != self.token_blur_level):
                # 需要计算局部tokens并回写
                device = get_device('auto')
                self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(self.model_type, device=f"cuda:{device}",pretrained=pretrain_map[self.model_type]['pretrained'])
                for param in self.vlmodel.parameters():
                    param.requires_grad = False
                self.vlmodel.eval()
                # 选取指定等级的模糊变换
                if isinstance(self.blur_transform, dict):
                    local_blur = self.blur_transform.get(self.token_blur_level, None)
                else:
                    local_blur = self.blur_transform
                self.img_local_tokens = self.ImageLocalTokens(self.loaded_data[0]['img'], blur_transform=local_blur)
                saved_features['img_local_tokens'] = self.img_local_tokens
                saved_features['tokens_blur_level'] = self.token_blur_level
                saved_features['img_local_tokens'] = self.img_local_tokens
                torch.save(saved_features, features_filename)
                del self.vlmodel
                torch.cuda.empty_cache()
                gc.collect()
        else:
            device = get_device('auto')
            self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(self.model_type, device=f"cuda:{device}",pretrained=pretrain_map[self.model_type]['pretrained'])
            for param in self.vlmodel.parameters():
                param.requires_grad = False
            self.vlmodel.eval()
            if self.config['data']['uncertainty_aware']:
                self.img_features = {}
                for tag in ['low','medium','high']:
                    self.img_features[tag] = self.ImageEncoder(self.loaded_data[0]['img'],self.blur_transform[tag])
                self.img_features['avg'] = {k: (sum(self.img_features[tag][k] for tag in ['low', 'medium', 'high']) / 3) for k in self.img_features['medium']}
            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            # 局部tokens采用指定模糊等级（默认 low/无模糊）
            if isinstance(self.blur_transform, dict):
                local_blur = self.blur_transform.get(self.token_blur_level, None)
            else:
                local_blur = self.blur_transform
            self.img_local_tokens = self.ImageLocalTokens(self.loaded_data[0]['img'], blur_transform=local_blur)
            # 保存时：若非不确定性模式，确保 img_features 为 {img_path: feature} 的扁平字典
            to_save_img_features = self.img_features
            if not self.config['data']['uncertainty_aware'] and isinstance(self.img_features, dict):
                # 已是扁平结构则直接保存
                # 如果误为多级结构，取 'medium' 或 'avg'
                keys = set(self.img_features.keys())
                if 'medium' in keys and isinstance(self.img_features['medium'], dict):
                    to_save_img_features = self.img_features['medium']
                elif 'avg' in keys and isinstance(self.img_features['avg'], dict):
                    to_save_img_features = self.img_features['avg']

            torch.save({
                'text_features': self.text_features,
                'img_features': to_save_img_features,
                'img_local_tokens': self.img_local_tokens,
                'tokens_blur_level': self.token_blur_level,
            }, features_filename)

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ',1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:,0]
            avg_data['img'] = loaded_data['img'][:,0]
            avg_data['text'] = loaded_data['text'][:,0]
                
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    @torch.no_grad()
    def ImageEncoder(self,images,blur_transform=None):
        if blur_transform == None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 128
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]

            device = next(self.vlmodel.parameters()).device
            ele = [self.process_transform(blur_transform(Image.open(os.path.join(self.data_dir,'../Image_set_Resize',img)).convert("RGB"))) for img in batch_images]

            image_inputs = torch.stack(ele).to(device)

            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]:image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict
    
    @torch.no_grad()
    def ImageLocalTokens(self, images, blur_transform=None):
        # 选择用于提取局部tokens的图像变换：若存在多级模糊，取'medium'或不模糊版本
        def default_blur(x):
            return x
        if blur_transform is not None:
            local_blur = blur_transform
        else:
            if isinstance(self.blur_transform, dict):
                desired = getattr(self, 'token_blur_level', 'low')
                local_blur = self.blur_transform.get(desired, default_blur)
            else:
                local_blur = self.blur_transform
        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 64
        result = {}

        # 通用hook注册，适配 open_clip / timm 多种结构
        def _get_blocks(visual):
            if hasattr(visual, 'trunk') and hasattr(visual.trunk, 'blocks'):
                return visual.trunk.blocks
            if hasattr(visual, 'blocks'):
                return visual.blocks
            if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
                return visual.transformer.resblocks
            return None

        def _flatten_tokens(x):
            # 支持 [B, N, C] 或 [B, C, H, W]
            if x.dim() == 3:
                # 去CLS: x[:, 1:, :]
                return x[:, 1:, :]
            elif x.dim() == 4:
                b, c, h, w = x.shape
                x = x.view(b, c, h*w).permute(0, 2, 1)
                return x
            else:
                return None

        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]
            device = next(self.vlmodel.parameters()).device
            ele = [self.process_transform(local_blur(Image.open(os.path.join(self.data_dir,'../Image_set_Resize',img)).convert("RGB"))) for img in batch_images]
            image_inputs = torch.stack(ele).to(device)

            # 注册hook
            visual = self.vlmodel.visual
            blocks = _get_blocks(visual)
            captures = {}
            hooks = []
            if blocks is not None and len(self.token_layers) > 0:
                L = len(blocks)
                layer_ids = [min(max(0, l-1), L-1) for l in self.token_layers]  # 层号转为0-index并截断
                for lid in layer_ids:
                    def _make_hook(name):
                        def hook_fn(module, inp, out):
                            captures[name] = out
                        return hook_fn
                    h = blocks[lid].register_forward_hook(_make_hook(f"layer_{lid}"))
                    hooks.append(h)

            # 前向运行（触发hook）
            _ = self.vlmodel.encode_image(image_inputs)

            # 取消hook
            for h in hooks:
                h.remove()

            # 聚合并选择Top-K tokens
            # 若未捕获，回退为空tokens
            if len(captures) == 0:
                feats = torch.zeros(image_inputs.shape[0], self.max_tokens_per_image, next(self.vlmodel.parameters()).shape[-1], device=image_inputs.device)
                feats = feats.float().cpu()
                for bi, img in enumerate(batch_images):
                    result[img] = feats[bi]
                continue

            # 将多个层的tokens拼接在一起： [B, Sum(L_i), C]
            concat_tokens = []
            for key in sorted(captures.keys()):
                x = captures[key]
                x_flat = _flatten_tokens(x)
                if x_flat is not None:
                    concat_tokens.append(x_flat)
            if len(concat_tokens) == 0:
                feats = torch.zeros(image_inputs.shape[0], self.max_tokens_per_image, next(self.vlmodel.parameters()).shape[-1], device=image_inputs.device)
                feats = feats.float().cpu()
                for bi, img in enumerate(batch_images):
                    result[img] = feats[bi]
                continue
            tokens = torch.cat(concat_tokens, dim=1)  # [B, K_all, C]

            # Top-K by L2 norm，并固定长度K（不足则零填充）
            with torch.no_grad():
                norms = torch.norm(tokens, dim=-1)  # [B, K_all]
                K = self.max_tokens_per_image
                topk_idx = torch.topk(norms, k=min(K, tokens.size(1)), dim=1).indices  # [B, k']
                gathered = torch.gather(tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))  # [B, k', C]
                if gathered.size(1) < K:
                    pad = torch.zeros(gathered.size(0), K - gathered.size(1), gathered.size(2), device=gathered.device)
                    gathered = torch.cat([gathered, pad], dim=1)
            gathered = gathered.float().cpu()

            for bi, img in enumerate(batch_images):
                result[img] = gathered[bi]
        return result
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text])
        device = next(self.vlmodel.parameters()).device
        text_inputs =  text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index//self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]

        img = 'None' #Image.open(os.path.join(self.data_dir,'../Image_set_Resize',img_path)).convert("RGB")
    
        match_label = self.match_label[index]
        
        if self.config['data']['uncertainty_aware']:
            if self.mode == 'train':
                if match_label==0:
                    tag='low'
                elif match_label==2:
                    tag='high'
                else:
                   tag='medium'
            else:
                tag='medium'
            img_features = self.img_features[tag][img_path]
            if self.return_multi_blurs:
                order = ['low','medium','high']
                img_features_multi = torch.stack([self.img_features[t][img_path] for t in order], dim=0)
        else:
            img_features = self.img_features[img_path]
            img_features_multi = None

        # 局部tokens固定长度 [K, D]
        if hasattr(self, 'img_local_tokens') and self.img_local_tokens is not None and img_path in self.img_local_tokens:
            img_local_tokens = self.img_local_tokens[img_path]
        else:
            # 若不存在，则回退为重复的全局特征（避免batch拼接失败），K次复制
            K = self.max_tokens_per_image
            img_local_tokens = img_features.unsqueeze(0).repeat(K, 1)

        text =  f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]
        
        sample  = {
            'idx': index,
            'eeg': eeg[:,self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'img_path': img_path,
            'img': img,
            'img_features': img_features,
            'img_local_tokens': img_local_tokens,  # [K, D]
            'text': text,
            'text_features': text_features,
            'session': session,
            'subject': subject,
            'eeg_mean': eeg_mean[:,self.timesteps[0]:self.timesteps[1]],
            'match_label': match_label,
        }
        if self.config['data']['uncertainty_aware'] and self.return_multi_blurs:
            sample['img_features_multi'] = img_features_multi
        return sample
    
    def __len__(self):
        return self.trial_all_subjects
    