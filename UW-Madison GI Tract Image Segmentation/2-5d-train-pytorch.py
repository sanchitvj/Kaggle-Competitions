import numpy as np
import pandas as pd
import random, datetime
from glob import glob
import os, shutil, sys
from tqdm import tqdm

tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold, GroupKFold

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

import timm
import pl_bolts

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.metrics.utils import get_mask_edges
from monai.metrics.utils import get_surface_distance

import rasterio
from joblib import Parallel, delayed

# For colored terminal text
from colorama import Fore, Back, Style

c_ = Fore.GREEN
sr_ = Style.RESET_ALL

import warnings

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import wandb

# sys.path.append("./segmentation_models.pytorch-master/")#segmentation_models_pytorch")
import segmentation_models_pytorch as smp


current_time = datetime.datetime.now()
subdir = str(current_time.day) + "-" + str(current_time.month) + "-" + str(current_time.year) + "__"
time_name = str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
current_exp_time = subdir + time_name


class CFG:
    seed = 111  # 1026
    num_workers = 8
    debug = False  # set debug=False for Full Training
    WANDB_NAME = "u_rgx6_320_bs32_e20"
    WT_NAME = "b0_320x320"
    exp_name = "timm-regnetx_006"
    model_name = "unet"
    shared_enc = False
    losses = "dice + bce + travesky"
    weights = "imagenet"  # ["imagenet", "noisy-student"] #
    backbone = "efficientnet-b7"  # ["timm-efficientnetv2-l-in21ft1k", "timm-efficientnet-b0"] # "timm-resnest200e" #
    attention_type = None  # 'scag' #'cbam' # 'scse'
    train_bs = 32
    valid_bs = train_bs * 2
    img_size = [320, 320]  # [320, 384] # [448, 512] [384, 480]
    num_slices = 3
    epochs = 20
    lr = 2e-3
    scheduler = "CosineAnnealingLR"  # "CosineAnnealingWarmRestarts"  # "LinearWarmupCosineAnnealingLR"
    min_lr = 1e-6
    warmup_start_lr = min_lr
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 2
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    folds = [0, 1, 2, 3, 4]  #
    num_classes = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


OUTPUT_DIR = f"./output/{current_exp_time}_{CFG.WANDB_NAME}"
os.makedirs(OUTPUT_DIR)


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("> SEEDING DONE")


set_seed(CFG.seed)

# mask_data_448x448x3
path_df = pd.DataFrame(glob("../data/mask_data_320x320x3/images/images/*"), columns=["image_path"])
# path_df = pd.DataFrame(glob('../data/mask25d_152_data/images/*'), columns=['image_path'])
path_df["mask_path"] = path_df.image_path.str.replace("image", "mask")
path_df["id"] = path_df.image_path.map(lambda x: x.split("/")[-1].replace(".npy", ""))
path_df.head()


df = pd.read_csv("../data/mask_data/train.csv")
# df['segmentation'] = df.segmentation.fillna('')
# df['rle_len'] = df.segmentation.map(len) # length of each rle mask

# df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
# df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

# df = df.drop(columns=['segmentation', 'class', 'rle_len'])
# df = df.groupby(['id']).head(1).reset_index(drop=True)
# df = df.merge(df2, on=['id'])
# df['empty'] = (df.rle_len==0) # empty masks

df = df.drop(columns=["image_path", "mask_path"])
df = df.merge(path_df, on=["id"])
# df.head()
# df.to_csv("test.csv")

# # Remove Faulty

fault1 = "case7_day0"
fault2 = "case81_day30"
df = df[~df["id"].str.contains(fault1) & ~df["id"].str.contains(fault2)].reset_index(drop=True)
# df.head()


# df['empty'].value_counts().plot.bar()


def id2mask(id_):
    idf = df[df["id"] == id_]
    wh = idf[["height", "width"]].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
        cdf = idf[idf["class"] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)


# def load_img(path, size=CFG.img_size):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     shape0 = np.array(img.shape[:2])
#     resize = np.array(size)
#     if np.any(shape0!=resize):
#         diff = resize - shape0
#         pad0 = diff[0]
#         pad1 = diff[1]
#         pady = [pad0//2, pad0//2 + pad0%2]
#         padx = [pad1//2, pad1//2 + pad1%2]
#         img = np.pad(img, [pady, padx])
#         img = img.reshape((*resize))
#     return img, shape0

# def load_imgs(img_paths, size=CFG.img_size):
#     imgs = np.zeros((*size, len(img_paths)), dtype=np.float32)
#     print(img_paths)
#     for i, img_path in enumerate(img_paths):
#         if i==0:
#             img, shape0 = load_img(img_path, size=size)
#         else:
#             img, _ = load_img(img_path, size=size)
#         img = img.astype('float32') # original is uint16
#         mx = np.max(img)
#         if mx:
#             img/=mx # scale image to [0, 1]
#         imgs[..., i]+=img
#     return imgs, shape0

########################## below is default #########################
def load_img(path):
    img = np.load(path)
    img = img.astype("float32")  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
# skf = GroupKFold(n_splits=CFG.n_fold)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["empty"], groups=df["case"])):
    df.loc[val_idx, "fold"] = fold


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path"].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img)


data_transforms = {
    "train": A.Compose(
        [
            #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
            #             A.RandomResizedCrop(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    #                                 A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                ],
                p=0.3,
            ),
            #             A.OneOf(
            #                 [
            #                     #             A.IAASharpen(alpha=(0.1, 0.3), p=0.5),
            #                     #             A.CLAHE(p=0.8),
            #                     A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            #                     # GaussianBlur(blur_limit=3, p=0.5),
            #                     #             A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            #                     A.RandomGamma(p=0.8),
            #                 ],
            #                 p=0.5,
            #             ),
            A.CoarseDropout(
                max_holes=8,
                max_height=CFG.img_size[0] // 20,
                max_width=CFG.img_size[1] // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            #         A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ],
        p=1.0,
    ),
}


def prepare_loaders(fold, debug=False):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    if debug:
        train_df = train_df.head(32 * 5).query("empty==0")
        valid_df = valid_df.head(32 * 3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms["train"])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms["valid"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_bs if not debug else 20,
        num_workers=CFG.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_bs if not debug else 20,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_loader, valid_loader


train_loader, valid_loader = prepare_loaders(fold=0, debug=True)


imgs, msks = next(iter(train_loader))
imgs.size(), msks.size()


import gc

gc.collect()


class model_segmentation(nn.Module):
    def __init__(self):
        super(model_segmentation, self).__init__()

        self.model_unet = smp.UnetPlusPlus(
            encoder_name=CFG.backbone,
            encoder_weights=CFG.weights,
            in_channels=3,
            classes=CFG.num_classes,
            activation=None,
        )

        self.model_fpn = smp.FPN(
            encoder_name=CFG.backbone,
            encoder_weights=CFG.weights,
            in_channels=3,
            classes=CFG.num_classes,
            activation=None,
        )

    def forward(self, x):

        global_features_unet = self.model_unet.encoder(x)
        # print(global_features[0].size())
        #         global_features_fpn = self.model_fpn.encoder(x)
        #         wt = torch.tensor([0.5], dtype=torch.float16).to(CFG.device)
        #         print(global_features_fpn[0].size(), global_features_fpn[1].size(), global_features_fpn[2].size(), global_features_fpn[3].size(), global_features_fpn[4].size(), global_features_fpn[5].size())

        global_features = global_features_unet  # []
        #         for i in range(6):

        #             if global_features_unet[i].size(dim=1) != global_features_fpn[i].size(dim=1):
        #                 if global_features_unet[i].size(dim=1) > global_features_fpn[i].size(dim=1):
        # #                     global_features_fpn[i] = F.pad(global_features_fpn[i], pad=(0, 1, 0, 0), mode='constant', value=0)
        #                     pad_size1, pad_size3, pad_size4 = global_features_unet[i].size(dim=0), global_features_unet[i].size(dim=2), global_features_unet[i].size(dim=3)
        #                     pad_size2 = global_features_unet[i].size(dim=1) - global_features_fpn[i].size(dim=1)
        #                     padded = torch.zeros(pad_size1, pad_size2, pad_size3, pad_size4).to(CFG.device)
        #                     global_features_fpn[i] = torch.cat([global_features_fpn[i], padded], dim=1)
        #                 else:
        #                     pad_size1, pad_size3, pad_size4 = global_features_unet[i].size(dim=0), global_features_unet[i].size(dim=2), global_features_unet[i].size(dim=3)
        #                     pad_size2 = global_features_fpn[i].size(dim=1) - global_features_unet[i].size(dim=1)
        #                     padded = torch.zeros(pad_size1, pad_size2, pad_size3, pad_size4).to(CFG.device)
        #                     global_features_unet[i] = torch.cat([global_features_unet[i], padded], dim=1)

        #             global_features.append(torch.mul(wt, global_features_unet[i]) + torch.mul(wt, global_features_fpn[i]))

        # IndexError: tuple index out of range -> *global_features
        seg_feature_unet = self.model_unet.decoder(*global_features)
        seg_feature_fpn = self.model_fpn.decoder(*global_features)

        out_u = self.model_unet.segmentation_head(seg_feature_unet)
        out_f = self.model_fpn.segmentation_head(seg_feature_fpn)

        return 0.65 * out_u + 0.35 * out_f


def build_model():

    if CFG.shared_enc:
        model = model_segmentation()
    else:
        model = smp.Unet(  # DeepLabV3Plus, UnetPlusPlus, Unet
            encoder_name=CFG.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=CFG.weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=CFG.num_slices,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
            activation=None,
            decoder_attention_type=CFG.attention_type,
        )
    model.to(CFG.device)
    return model


def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# def compute_competition_metric(preds: np.ndarray, targets: np.ndarray) -> float:
#     dice_ = compute_dice(preds, targets)
#     hd_dist_ = compute_hd_dist(preds, targets)
#     return 0.4 * dice_ + 0.6 * hd_dist_, dice_, hd_dist_


# Slightly adapted from https://www.kaggle.com/code/carnozhao?scriptVersionId=93589877&cellId=2
def compute_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    preds = preds.astype(np.uint8)
    targets = targets.astype(np.uint8)

    I = (targets & preds).sum((2, 3))  # noqa: E741
    U = (targets | preds).sum((2, 3))  # noqa: E741

    return np.mean((2 * I / (U + I + 1) + (U == 0)).mean(1))


def compute_hd_dist(preds: np.ndarray, targets: np.ndarray) -> float:
    return 1 - np.mean([hd_dist_batch(preds[:, i, ...], targets[:, i, ...]) for i in range(3)])


def hd_dist_batch(preds: np.ndarray, targets: np.ndarray) -> float:
    return np.mean([hd_dist(pred, target) for pred, target in zip(preds, targets)])


# From https://www.kaggle.com/code/yiheng?scriptVersionId=93883465&cellId=4
def hd_dist(pred: np.ndarray, target: np.ndarray) -> float:
    if np.all(pred == target):
        return 0.0

    edges_pred, edges_gt = get_mask_edges(pred, target)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")

    if surface_distance.shape == (0,):
        return 0.0

    dist = surface_distance.max()
    max_dist = np.sqrt(np.sum(np.array(pred.shape) ** 2))

    if dist > max_dist:
        return 1.0

    return dist / max_dist


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


JaccardLoss = smp.losses.JaccardLoss(mode="multilabel")
DiceLoss = smp.losses.DiceLoss(mode="multilabel")
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
# LovaszLoss = smp.losses.LovaszLoss(mode="multilabel", per_image=False) # don't use
TverskyLoss = smp.losses.TverskyLoss(mode="multilabel", log_loss=False)


def criterion(y_pred, y_true):
    return (
        0.4 * DiceLoss(y_pred, y_true)
        + 0.3 * BCELoss(y_pred, y_true)
        + 0.3 * TverskyLoss(y_pred, y_true)
        #         + 0.2 * JaccardLoss(y_pred, y_true)
    )


#     return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
#     return DiceLoss(y_pred, y_true)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train ")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            train_loss=f"{epoch_loss:0.4f}",
            lr=f"{current_lr:0.5f}",
            gpu_mem=f"{mem:0.2f} GB",
        )
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, thr=0.5):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores, y_preds, y_true = [], [], []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        #         y_pred = (y_pred > thr).cpu().detach().to(torch.float32)
        #         y_preds.append(y_pred), y_true.append(masks.cpu().detach().to(torch.float32))

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(
            valid_loss=f"{epoch_loss:0.4f}",
            lr=f"{current_lr:0.5f}",
            gpu_memory=f"{mem:0.2f} GB",
        )
    val_scores = np.mean(val_scores, axis=0)
    #     y_preds_ = torch.cat(y_preds).numpy()
    #     y_true_ = torch.cat(y_true).numpy()
    #     dice_ = compute_dice(y_preds_, y_true_)
    #     hd_dist_ = compute_hd_dist(y_preds_, y_true_)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores  # , dice_, hd_dist_


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_cv = -np.inf
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}", end="")

        # scheduler.step(epoch)
        train_loss = train_one_epoch(
            model, optimizer, scheduler, dataloader=train_loader, device=CFG.device, epoch=epoch
        )

        #         val_loss, val_scores, dice_, hd_dist_ = valid_one_epoch(model, valid_loader, device=CFG.device, epoch=epoch)
        val_loss, val_scores = valid_one_epoch(model, valid_loader, device=CFG.device, epoch=epoch)
        val_dice, val_jaccard = val_scores
        #         cv = 0.6 * hd_dist_ + 0.4 * dice_
        #         avg_dice_cv = 0.6 * hd_dist_ + 0.4 * ((dice_ + val_dice) / 2)

        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        history["Valid Dice"].append(val_dice)
        history["Valid Jaccard"].append(val_jaccard)
        #         history["Valid Carno Dice"].append(dice_)
        #         history["Valid HD"].append(hd_dist_)
        #         history["Competition CV"].append(cv)
        #         history["Average dice CV"].append(avg_dice_cv)

        # Log the metrics
        wandb.log(
            {
                "Train Loss": train_loss,
                "Valid Loss": val_loss,
                "Valid Dice": val_dice,
                "Valid Jaccard": val_jaccard,
                #                 "Valid Carno Dice": dice_,
                #                 "Valid HD": hd_dist_,
                #                 "Competition CV": cv,
                #                 "Average dice CV": avg_dice_cv,
                "LR": scheduler.get_lr()[0],  # .get_last_lr()[0],
            }
        )

        print(
            f"Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}"
        )  # | Competition CV: {cv:0.4f} | Average dice CV: {avg_dice_cv:0.4f}"
        #         )

        # deep copy the model
        #         if cv >= best_cv:
        if val_dice >= best_dice:
            #             print(f"{c_}Valid Score Improved ({best_cv:0.4f} ---> {cv:0.4f})")
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            #             best_cv = cv
            #             best_avg_dice_cv = avg_dice_cv
            #             best_hd = hd_dist_
            #             best_carno_dice = dice_
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            #             run.summary["Best Competition CV"] = best_cv
            #             run.summary["Best Average dice CV"] = best_avg_dice_cv
            #             run.summary["Best HD"] = best_hd
            #             run.summary["Best Carno Dice"] = best_carno_dice
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            #             PATH = f"best_epoch-{fold:02d}.bin"
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{CFG.WT_NAME}_model_fold{fold}.bin")
            torch.save(
                {
                    "epoch": epoch,
                    #                 'model': model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{OUTPUT_DIR}/{CFG.WT_NAME}_other_fold{fold}.bin",
            )
            # Save a model file from the current directory
            #             wandb.save(PATH)
            print(f"Model Saved{sr_}")

        #         last_model_wts = copy.deepcopy(model.state_dict())
        #         PATH = f"last_epoch-{fold:02d}.bin"
        #         torch.save(model.state_dict(), PATH)

        print()
        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Score: {:.4f}".format(best_cv))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CFG.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, eta_min=CFG.min_lr)
    elif CFG.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=7,
            threshold=0.0001,
            min_lr=CFG.min_lr,
        )
    elif CFG.scheduler == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == "LinearWarmupCosineAnnealingLR":
        scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(
            optimizer,
            CFG.warmup_epochs,
            CFG.epochs,
            warmup_start_lr=CFG.warmup_start_lr,
            eta_min=CFG.min_lr,
            last_epoch=-1,
        )
    elif CFG.scheduler == None:
        return None

    return scheduler


model = build_model()
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = fetch_scheduler(optimizer)

if __name__ == "__main__":

    for fold in CFG.folds:
        print(f"#" * 15)
        print(f"### Fold: {fold}")
        print(f"#" * 15)
        run = wandb.init(
            project="UWMGI",
            config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
            name=f"FOLD{fold}_{CFG.WANDB_NAME}",
        )
        train_loader, valid_loader = prepare_loaders(fold=fold, debug=CFG.debug)
        model = build_model()
        optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        scheduler = fetch_scheduler(optimizer)
        model, history = run_training(model, optimizer, scheduler, device=CFG.device, num_epochs=CFG.epochs)
        run.finish()
