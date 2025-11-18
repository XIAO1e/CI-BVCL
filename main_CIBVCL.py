import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
import time
from torch.optim import AdamW, Adam, SGD
import numpy as np
from collections import Counter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader, random_split, Dataset
from pytorch_lightning.callbacks import Callback

# user libs
from base.data_eeg import load_eeg_data
from base.data_meg import load_meg_data
from base.utils import update_config , ClipLoss, instantiate_from_config, get_device


device = get_device('auto')


def load_model(config, train_loader, test_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)

    pl_model = PLModel(model, config, train_loader, test_loader)
    return pl_model


def fourier_augment_amplitude(x: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Amplitude interpolation in frequency domain on EEG time-series.

    x: [B, C, T]
    return: [B, C, T]
    """
    X = torch.fft.fft(x, dim=-1)
    amp, ph = torch.abs(X), torch.angle(X)
    idx = torch.randperm(x.size(0), device=x.device)
    amp_rand = amp[idx]
    amp_aug = (1.0 - tau) * amp + tau * amp_rand
    X_aug = amp_aug * torch.exp(1j * ph)
    x_aug = torch.fft.ifft(X_aug, dim=-1).real
    return x_aug


def fourier_augment_phase(x: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Phase interpolation in frequency domain on EEG time-series.

    x: [B, C, T]
    return: [B, C, T]
    """
    X = torch.fft.fft(x, dim=-1)
    amp, ph = torch.abs(X), torch.angle(X)
    idx = torch.randperm(x.size(0), device=x.device)
    ph_rand = ph[idx]
    ph_aug = (1.0 - tau) * ph + tau * ph_rand
    X_aug = amp * torch.exp(1j * ph_aug)
    x_aug = torch.fft.ifft(X_aug, dim=-1).real
    return x_aug


def time_shift_augment(x: torch.Tensor, max_frac: float = 0.02) -> torch.Tensor:
    """Time-domain small shift (lower compute than FFT-based augmentation).

    Each sample is circularly shifted along the time dimension by an amount
    sampled uniformly from [-max_frac * T, max_frac * T].

    x: [B, C, T]
    return: [B, C, T]
    """
    if max_frac <= 0:
        return x
    T = x.size(-1)
    max_shift = max(1, int(max_frac * T))
    if max_shift <= 0:
        return x
    # per-sample random shifts in [-max_shift, max_shift]
    shifts = torch.randint(low=-max_shift, high=max_shift + 1, size=(x.size(0),), device=x.device)
    # apply roll per sample to avoid unnecessary copies
    rolled = [torch.roll(x[i], shifts=int(shifts[i].item()), dims=-1) for i in range(x.size(0))]
    return torch.stack(rolled, dim=0)


def amplitude_eq_augment(x: torch.Tensor, strength: float = 0.2, num_knots: int = 8) -> torch.Tensor:
    """Random smooth equalization on amplitude spectrum with symmetric gain.

    - Builds a smooth gain curve g(f) with piecewise-linear interpolation from
      num_knots random values in [1-strength, 1+strength].
    - Enforces symmetry g[k] = g[N-k] to preserve conjugate symmetry for real signals.
    - Applies gain on amplitude, keeps phase intact, then IFFT.

    x: [B, C, T]
    return: [B, C, T]
    """
    if strength <= 0 or num_knots <= 1:
        return x

    B, C, T = x.shape
    # FFT
    X = torch.fft.fft(x, dim=-1)
    amp, ph = torch.abs(X), torch.angle(X)

    # Prepare knot positions (uniform along frequency bins)
    knot_positions = torch.linspace(0, T - 1, steps=num_knots, device=x.device)

    def _interp_gain(batch_size: int) -> torch.Tensor:
        # Random knot gains in [1-strength, 1+strength]
        low = 1.0 - strength
        high = 1.0 + strength
        knot_gains = low + (high - low) * torch.rand(batch_size, num_knots, device=x.device)
        # Interpolate to length T (per-sample piecewise linear)
        # Use linear interpolation by projecting positions
        all_idx = torch.arange(T, device=x.device).float().unsqueeze(0)  # [1, T]
        # For each interval between knots, compute weights
        # Vectorized linear interpolation via searching left/right knot indices
        # Compute fractional position t in [0, num_knots-1]
        t = (all_idx * (num_knots - 1)) / (T - 1 + 1e-8)  # [1, T]
        left_idx = torch.clamp(t.floor().long(), 0, num_knots - 1)
        right_idx = torch.clamp(left_idx + 1, 0, num_knots - 1)
        frac = (t - left_idx.float())  # [1, T]
        left_val = knot_gains.gather(1, left_idx.expand(batch_size, -1))
        right_val = knot_gains.gather(1, right_idx.expand(batch_size, -1))
        gains = left_val * (1.0 - frac) + right_val * frac  # [B, T]
        return gains

    gains = _interp_gain(B)  # [B, T]
    # Enforce symmetry to preserve real-signal property when applied on amplitude
    gains_sym = 0.5 * (gains + torch.flip(gains, dims=[1]))  # [B, T]
    gains_sym = gains_sym.clamp(min=1.0 - strength, max=1.0 + strength)
    gains_sym = gains_sym.unsqueeze(1)  # [B, 1, T]

    amp_aug = amp * gains_sym
    X_aug = amp_aug * torch.exp(1j * ph)
    x_aug = torch.fft.ifft(X_aug, dim=-1).real
    return x_aug


def augment_image_features(img_features: torch.Tensor, augment_type: str = 'noise', tau: float = 0.5) -> torch.Tensor:
    """Augment image features with different types of perturbations.

    img_features: [B, D] - image feature embeddings
    augment_type: 'noise'|'dropout'|'scale'|'none'
    return: [B, D]
    """
    if augment_type == 'none':
        return img_features

    if augment_type == 'noise':
        # Add Gaussian noise
        noise = torch.randn_like(img_features) * tau
        return img_features + noise

    elif augment_type == 'dropout':
        # Random feature dropout
        mask = torch.rand_like(img_features) > tau
        return img_features * mask

    elif augment_type == 'scale':
        # Random scaling
        scale_factor = torch.rand(img_features.size(0), 1, device=img_features.device) * 2 * tau + (1 - tau)
        return img_features * scale_factor

    else:
        return img_features


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader, test_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.criterion = ClipLoss()

        self.all_predicted_classes = []
        self.all_true_labels = []

        self.z_dim = int(self.config['z_dim'])

        # CI-BVCL params（容错：若被 CLI 的 None 覆盖，则回退到默认值）
        def _cfg_get(key, default):
            val = self.config.get(key, default)
            return default if val is None else val

        self.w_causal = float(_cfg_get('w_causal', 0.1))
        aug_type_val = _cfg_get('augment_type', 'phase')
        # 'phase'|'amplitude'|'both'|'none'|'time_shift'|'amplitude_eq'
        self.augment_type = aug_type_val if isinstance(aug_type_val, str) else 'phase'
        self.augment_tau = float(_cfg_get('augment_tau', 0.5))
        # New augmentation params
        self.time_shift_frac = float(_cfg_get('time_shift_frac', 0.02))
        self.eq_strength = float(_cfg_get('eq_strength', 0.2))
        self.eq_num_knots = int(_cfg_get('eq_num_knots', 8))
        self.causal_temperature = float(_cfg_get('causal_temperature', 0.07))

        # Image augmentation params
        img_aug_type_val = _cfg_get('img_augment_type', 'noise')
        self.img_augment_type = img_aug_type_val if isinstance(img_aug_type_val, str) else 'noise'  # 'noise'|'dropout'|'scale'|'none'
        self.img_augment_tau = float(_cfg_get('img_augment_tau', 0.1))

        # Image projector
        self.img_projector = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.GELU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.LayerNorm(self.z_dim)
        )

    def _make_augmented_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        # eeg: [B, C, T]
        if self.augment_type == 'none':
            return eeg
        if self.augment_type == 'phase':
            return fourier_augment_phase(eeg, tau=self.augment_tau)
        if self.augment_type == 'amplitude':
            return fourier_augment_amplitude(eeg, tau=self.augment_tau)
        if self.augment_type == 'both':
            y = fourier_augment_amplitude(eeg, tau=self.augment_tau)
            y = fourier_augment_phase(y, tau=self.augment_tau)
            return y
        if self.augment_type == 'time_shift':
            return time_shift_augment(eeg, max_frac=self.time_shift_frac)
        if self.augment_type == 'amplitude_eq':
            return amplitude_eq_augment(eeg, strength=self.eq_strength, num_knots=self.eq_num_knots)
        return eeg

    def _make_augmented_image(self, img_features: torch.Tensor) -> torch.Tensor:
        # img_features: [B, D]
        return augment_image_features(img_features, augment_type=self.img_augment_type, tau=self.img_augment_tau)

    @staticmethod
    def _info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        # Normalize
        a = anchor / (anchor.norm(dim=-1, keepdim=True) + 1e-8)
        p = positive / (positive.norm(dim=-1, keepdim=True) + 1e-8)
        logits = (a @ p.t()) / temperature
        labels = torch.arange(a.size(0), device=a.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(self, batch):
        eeg = batch['eeg']             # [B, C, T]
        img_z  = batch['img_features'] # [B, z_dim]

        eeg_z = self.brain(eeg)        # [B, z_dim]
        img_z = self.img_projector(img_z)  # Apply image projector
        img_z = img_z / img_z.norm(dim=-1, keepdim=True)

        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        sup_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        causal_loss = torch.zeros((), device=eeg.device)

        # Causal contrastive via Fourier augmentation on EEG time series
        if self.w_causal > 0.0 and self.augment_type != 'none':
            with torch.no_grad():
                eeg_aug = self._make_augmented_eeg(eeg)
            eeg_z_aug = self.brain(eeg_aug)
            eeg_causal_loss = self._info_nce(eeg_z, eeg_z_aug, temperature=self.causal_temperature)
            causal_loss = causal_loss + eeg_causal_loss

        # Causal contrastive via image feature augmentation
        if self.w_causal > 0.0 and self.img_augment_type != 'none':
            with torch.no_grad():
                img_aug = self._make_augmented_image(batch['img_features'])
            img_z_aug = self.img_projector(img_aug)
            img_z_aug = img_z_aug / img_z_aug.norm(dim=-1, keepdim=True)
            img_causal_loss = self._info_nce(img_z, img_z_aug, temperature=self.causal_temperature)
            causal_loss = causal_loss + img_causal_loss

        loss = sup_loss + self.w_causal * causal_loss
        return eeg_z, img_z, loss, sup_loss, causal_loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, sup_loss, causal_loss = self(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('train_sup_loss', sup_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('train_causal_loss', causal_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss

    def on_train_epoch_end(self):
        if len(self.all_predicted_classes) == 0:
            return
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, sup_loss, causal_loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('val_sup_loss', sup_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('val_causal_loss', causal_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        if len(self.all_predicted_classes) == 0:
            return
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss, sup_loss, causal_loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = (eeg_z @ img_z.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        if len(self.all_predicted_classes) == 0:
            return
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'], weight_decay=1e-4)
        return [optimizer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eeg/CI-BVCL.yaml", help="path to config")
    parser.add_argument("--dataset", type=str, default="eeg", choices=["eeg", "meg"], help="dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=str, default='sub-08')
    parser.add_argument("--exp_setting", type=str, default='intra-subject')
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--brain_backbone", type=str, default='CNNTransformerEEG')
    parser.add_argument("--vision_backbone", type=str, default='ViT-H-14')
    parser.add_argument("--c", type=int, default=6)
    parser.add_argument("--selected_ch", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)

    # CI-BVCL specific
    parser.add_argument("--augment_type", type=str, default=None, choices=[None, 'phase', 'amplitude', 'both', 'none', 'time_shift', 'amplitude_eq'])
    parser.add_argument("--augment_tau", type=float, default=None)
    parser.add_argument("--time_shift_frac", type=float, default=None)
    parser.add_argument("--eq_strength", type=float, default=None)
    parser.add_argument("--eq_num_knots", type=int, default=None)
    parser.add_argument("--w_causal", type=float, default=None)
    parser.add_argument("--causal_temperature", type=float, default=None)
    # Image augmentation parameters
    parser.add_argument("--img_augment_type", type=str, default=None, choices=[None, 'noise', 'dropout', 'scale', 'none'])
    parser.add_argument("--img_augment_tau", type=float, default=None)

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    if opt.name is not None:
        config['name'] = opt.name
    config['data']['subjects'] = [opt.subjects]

    # Override selected_ch if provided
    if opt.selected_ch is not None:
        sel = opt.selected_ch.strip()
        if sel.lower() == 'none':
            config['data']['selected_ch'] = "None"
        else:
            parsed = [s.strip() for s in sel.split(',') if s.strip()]
            config['data']['selected_ch'] = parsed

    # Override batch size and num_workers
    if opt.batch_size is not None:
        config['data']['train_batch_size'] = opt.batch_size
    if opt.num_workers is not None:
        config['data']['num_workers'] = opt.num_workers

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }
    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    print(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    # Name fallback
    if ('name' not in config) or (config['name'] is None) or (str(config['name']).strip() == ''):
        ds = str(config.get('dataset', 'exp'))
        setting = str(config.get('exp_setting', 'run'))
        brain = str(opt.brain_backbone or 'brain')
        vision = str(opt.vision_backbone or 'vision')
        config['name'] = f"{ds}_{setting}_CI-BVCL_{brain}_{vision}"
    version = f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}"
    run_dir = os.path.join(config['save_dir'], config['name'], version)
    os.makedirs(run_dir, exist_ok=True)
    logger = TensorBoardLogger(config['save_log_dir'], name=config['name'], version=version)
    os.makedirs(logger.log_dir, exist_ok=True)
    # Save config
    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))
    shutil.copy(opt.config, os.path.join(run_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)

    # In intra-subject, split val from train
    if config['exp_setting'] == 'intra-subject':
        full_train_dataset = train_loader.dataset
        g = torch.Generator().manual_seed(int(config.get('seed', 0)))
        val_ratio = float(config['data'].get('val_ratio', 0.1))
        val_len = max(1, int(len(full_train_dataset) * val_ratio))
        train_len = len(full_train_dataset) - val_len

        train_subset, val_subset = random_split(full_train_dataset, [train_len, val_len], generator=g)

        class IndexedSubset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                sample = self.dataset[self.indices[i]]
                sample['idx'] = i
                return sample

        default_num_workers = getattr(train_loader, 'num_workers', 8)
        configured_num_workers = config['data'].get('num_workers', default_num_workers)
        train_dataset = IndexedSubset(full_train_dataset, train_subset.indices)
        val_dataset = IndexedSubset(full_train_dataset, val_subset.indices)

        train_loader = DataLoader(train_dataset, batch_size=config['data']['train_batch_size'], shuffle=True, drop_last=False, num_workers=configured_num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['val_batch_size'], shuffle=False, drop_last=False, num_workers=configured_num_workers, pin_memory=True)

    # Auto-detect channels and override c_num
    try:
        eeg_sample = train_loader.dataset[0]['eeg']
        detected_c_num = int(eeg_sample.shape[0]) if eeg_sample.dim() == 2 else int(eeg_sample.shape[1])
    except Exception:
        batch = next(iter(train_loader))
        eeg_sample = batch['eeg']
        detected_c_num = int(eeg_sample.shape[1])
    if 'models' in config and 'brain' in config['models'] and 'params' in config['models']['brain']:
        config['models']['brain']['params']['c_num'] = detected_c_num

    print(f"train num: {len(train_loader.dataset)}, val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}")
    pl_model = load_model(config, train_loader, test_loader)
    
    # 初始化模型参数：运行一次虚拟前向传播
    print("Initializing model parameters...")
    with torch.no_grad():
        # 获取一个小的批次进行初始化
        sample_batch = next(iter(train_loader))
        # 只保留模型需要的前向传播的键
        mini_batch = {
            'eeg': sample_batch['eeg'][:2],  # 只取2个样本
            'img_features': sample_batch['img_features'][:2]
        }

        # 运行前向传播来初始化所有参数
        _ = pl_model(mini_batch)
    print("Model parameters initialized successfully.")

    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    # Use validation metrics for both intra-subject and inter-subject settings
    # This ensures we select models that generalize well to unseen data
    checkpoint_callback = ModelCheckpoint(
        save_last=True, 
        monitor='val_top1_acc', 
        mode='max', 
        save_top_k=1, 
        dirpath=os.path.join(run_dir, 'checkpoints'), 
        filename='best-{val_top1_acc:.4f}-{epoch:02d}'
    )
    
    class EnsureDirExistsCallback(Callback):
        def __init__(self, path):
            super().__init__()
            self.path = path
        def _ensure(self):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass
        def on_train_epoch_start(self, trainer, pl_module):
            self._ensure()
        def on_validation_epoch_start(self, trainer, pl_module):
            self._ensure()
        def on_train_epoch_end(self, trainer, pl_module):
            self._ensure()
        def on_validation_epoch_end(self, trainer, pl_module):
            self._ensure()
    early_stop_callback = EarlyStopping(
        monitor='val_top1_acc',
        min_delta=0.001,
        patience=config.patience,
        verbose=False,
        mode='max'
    )

    ensure_dir_cb = EnsureDirExistsCallback(os.path.join(run_dir, 'checkpoints'))
    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=False), callbacks=[ensure_dir_cb, early_stop_callback, checkpoint_callback], max_epochs=config['train']['epoch'], devices=[device], accelerator='cuda', logger=logger)
    print(trainer.logger.log_dir)

    ckpt_path = None
    train_start_ts = time.time()
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    train_time_seconds = time.time() - train_start_ts

    test_start_ts = time.time()
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    test_time_seconds = time.time() - test_start_ts

    to_write = {
        'test_results': test_results,
        'train_time_seconds': train_time_seconds,
        'test_time_seconds': test_time_seconds,
        'subject': config['data']['subjects'][0] if isinstance(config['data']['subjects'], list) and len(config['data']['subjects']) > 0 else None,
        'exp_name': config['name'],
        'version': version
    }
    with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
        json.dump(to_write, f, indent=4)


if __name__ == "__main__":
    main()


