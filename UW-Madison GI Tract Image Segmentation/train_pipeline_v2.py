import numpy as np
import pandas as pd
import random, datetime
from glob import glob
import os, shutil
from tqdm import tqdm

tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
import warnings

warnings.filterwarnings("ignore")

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda import amp
from scipy.spatial.distance import directed_hausdorff

import timm
from timm.optim.lookahead import Lookahead

# Albumentations for augmentations
import albumentations as A

# from albumentations.pytorch import ToTensorV2

# import rasterio
from joblib import Parallel, delayed

# For colored terminal text
from colorama import Fore, Back, Style

c_ = Fore.GREEN
sr_ = Style.RESET_ALL


# For descriptive error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import segmentation_models_pytorch as smp

import wandb

current_time = datetime.datetime.now()
subdir = str(current_time.day) + "-" + str(current_time.month) + "-" + str(current_time.year) + "__"
time_name = str(current_time.hour) + "_" + str(current_time.minute) + "_" + str(current_time.second)
current_exp_time = subdir + time_name


class CFG:
    seed = 101
    num_workers = 8
    debug = False  # set debug=False for Full Training
    WANDB_NAME = "u_rgx6_320_bs32_e20"
    WT_NAME = "b0_320x320"
    exp_name = "timm-regnetx_006"
    model_name = "unet"
    shared_enc = False
    weights = "imagenet"  # ["imagenet", "noisy-student"] #
    backbone = "efficientnet-b7"  # ["timm-efficientnetv2-l-in21ft1k", "timm-efficientnet-b0"] # "timm-resnest200e" #
    attention_type = None  # 'scag' #'cbam' # 'scse'
    train_bs = 32
    valid_bs = train_bs * 2
    img_size = [224, 320]  # [320, 384] # [448, 512] [384, 480]
    num_slices = 3
    epochs = 20
    lr = 1e-3
    scheduler = "CosineAnnealingLR"  # "CosineAnnealingWarmRestarts"  # "LinearWarmupCosineAnnealingLR"
    min_lr = 1e-6
    warmup_start_lr = min_lr
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 2
    wd = 1e-5
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    foldwise = True  # foldwise->True, full data->False
    folds = [0, 1, 2, 3, 4]  #
    num_classes = 3
    # fold = -1
    description = "timm-efficientnet-b4 imagenet 15 epochs cropped casewise"
    scaling_factor = 16384.0
    loss_info = {"Dice": 1.0, "Focal": 0.3}
    lookahead_k = 10
    lookahead_alpha = 0.25
    warmup_factor = 10
    focal_alpha = None
    focal_gamma = 2.0
    casewise_scaling = True
    cropped_analysis = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name()


OUTPUT_DIR = f"./output/{current_exp_time}_{CFG.WANDB_NAME}"
os.makedirs(OUTPUT_DIR)

#  se_resnext50_32x4d   0.9012  OOM
#  timm-res2next50      0.9063  OOM
#  timm-resnest50d      0.9088  OOM
#  resnext50_32x4d      0.9000  OOM

# timm-efficientnet-b2  15.86 Completed
# se_resnext50_32x4d    15.95
# timm-res2next50       15.20
# timm-resnest50d       15.34 34
# resnext50_32x4d       14.57

# Final Models
# 1. timm-efficientnet-b0 imagenet
# 2. timm-efficientnet-b1 imagenet
# 3. timm-efficientnet-b2 noisy-student
# 4. timm-res2next50
# 5. timm-resnest50d
# 6. resnext50_32x4d


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


# ## Data_Prepare

df = pd.read_csv("/content/drive/MyDrive/uwm/2.5d_data/cropped_splits.csv")
df["mask_path"] = "/content/uwm_mask/" + df["id"] + ".npy"
df["image_path"] = "/content/uwm/" + df["id"] + ".npy"
print(df.shape)
df = df[~((df.case == 7) & (df.day == 0))]
df = df[~((df.case == 81) & (df.day == 30))]
print(df.shape)

df.col_diff.unique(), df.row_diff.unique()

df["which_segs"].value_counts()

df["weights"] = 0.75
df.loc[df["which_segs"] == 0, "weights"] = 0.25
df.weights.value_counts()


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


def load_img(path, scaling_factor=CFG.scaling_factor, pos_info=None):
    img = np.load(path)
    img = img.astype("float32")  # original is uint16
    img /= float(scaling_factor)  # scale image to [0, 1]
    if pos_info is not None:
        # print(f"img -- {path}", img[int(pos_info['r1']):int(pos_info['r2']), int(pos_info['c1']):int(pos_info['c2']), :].shape)
        return img[int(pos_info["r1"]) : int(pos_info["r2"]), int(pos_info["c1"]) : int(pos_info["c2"]), :]
    else:
        return img


def load_msk(path, pos_info=None):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    if pos_info is not None:
        # print(f"mask -- {path}", msk[int(pos_info['r1']):int(pos_info['r2']), int(pos_info['c1']):int(pos_info['c2']), :].shape)
        return msk[int(pos_info["r1"]) : int(pos_info["r2"]), int(pos_info["c1"]) : int(pos_info["c2"]), :]
    else:
        return msk


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap="bone")

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [
            Rectangle((0, 0), 1, 1, color=_c) for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles, labels)
    plt.axis("off")


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        # self.df = df
        self.len = len(df)
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.msk_paths = df["mask_path"].tolist()
        self.scaling_factor = df["max_max"].tolist()
        self.transforms = transforms
        if CFG.cropped_analysis:
            self.r1 = df["n_row_1"].tolist()
            self.r2 = df["n_row_2"].tolist()
            self.c1 = df["n_col_1"].tolist()
            self.c2 = df["n_col_2"].tolist()
        else:
            self.r1 = None
            self.r2 = None
            self.c1 = None
            self.c2 = None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        ## Cropped Analysis
        if CFG.cropped_analysis:
            pos_info = {"r1": self.r1[index], "r2": self.r2[index], "c1": self.c1[index], "c2": self.c2[index]}
        else:
            pos_info = None

        ## Casewise Scaling
        if CFG.casewise_scaling:
            img = load_img(img_path, scaling_factor=self.scaling_factor[index], pos_info=pos_info)
        else:
            img = load_img(img_path, pos_info=pos_info)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path, pos_info=pos_info)
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


# ## Transforms

data_transforms = {
    "train": A.Compose(
        [
            # Resizing Transforms
            # A.Blur(blur_limit=5, p=0.5),
            # A.PadIfNeeded(CFG.img_size[0], CFG.img_size[1], border_mode=cv2.BORDER_CONSTANT, value =0),
            # A.OneOf([A.RandomSizedCrop(min_max_height=(int(CFG.img_size[0]*0.9), int(CFG.img_size[0])), height=CFG.img_size[0], width=CFG.img_size[1], interpolation=cv2.INTER_NEAREST, p=1.0),
            #          A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST)], p=1.0),
            # Pixel Augmentations
            A.RandomBrightnessContrast(p=0.75),
            # A.RandomGamma(p=0.5),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # # Blur Augmentations
            # A.OneOf([A.Blur(blur_limit=5, p=1.0),
            #          A.MedianBlur(blur_limit=5, p=1.0),
            #          A.MotionBlur(p=1.0),], p=0.3),
            # Shift Rotate
            A.ShiftScaleRotate(
                shift_limit=15 / CFG.img_size[0],
                scale_limit=15 / CFG.img_size[0],
                rotate_limit=30,
                interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            # Grid Transforms
            A.OneOf(
                [
                    A.ElasticTransform(interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.GridDistortion(
                        distort_limit=0.05, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=1.0
                    ),
                ],
                p=0.25,
            ),
            # A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ],
        p=1.0,
    ),
    # "valid": A.Compose([A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),], p=1.0)
    "valid": A.Compose([], p=1.0),
}


CFG.augmentations = data_transforms


def prepare_loaders(fold, debug=False):

    if CFG.foldwise:
        train_df = df.query("fold!=@fold").reset_index(drop=True)
        valid_df = df.query("fold==@fold").reset_index(drop=True)
    else:
        train_df = df
        valid_df = df

    if debug:
        train_df = train_df.head(32 * 5)  # .query("empty==0")
        valid_df = valid_df.head(32 * 3)  # .query("empty==0")

    train_dataset = BuildDataset(train_df, transforms=data_transforms["train"])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms["valid"])

    sampler_weights = torch.from_numpy(train_df.weights.to_numpy())
    sampler = WeightedRandomSampler(sampler_weights.type("torch.DoubleTensor"), len(sampler_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_bs if not debug else 20,
        sampler=sampler,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.valid_bs if not debug else 20,
        num_workers=CFG.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, valid_loader


def plot_batch(imgs, msks, size=3):
    plt.figure(figsize=(5 * 5, 5))
    for idx in range(size):
        plt.subplot(1, 5, idx + 1)
        img = (
            imgs[
                idx,
            ]
            .permute((1, 2, 0))
            .numpy()
            * 255.0
        )
        img = img.astype("uint8")
        msk = (
            msks[
                idx,
            ]
            .permute((1, 2, 0))
            .numpy()
            * 255.0
        )
        show_img(img, msk)
    plt.tight_layout()
    plt.show()


# ### Metrics


loss_dict = {
    "Jaccard": smp.losses.JaccardLoss(mode="multilabel"),
    "Dice": smp.losses.DiceLoss(mode="multilabel"),
    "BCE": smp.losses.SoftBCEWithLogitsLoss(),
    "Focal": smp.losses.FocalLoss(alpha=CFG.focal_alpha, gamma=CFG.focal_gamma, mode="binary"),
    "Lovasz": smp.losses.LovaszLoss(mode="multilabel", per_image=False),
    "Tversky": smp.losses.TverskyLoss(mode="multilabel", log_loss=False),
}

for func_name, weight in CFG.loss_info.items():
    print(loss_dict[func_name])


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    if dim == (2, 3):
        mean_dim = (1, 0)
    elif dim == (1, 2):
        mean_dim = 0
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=mean_dim)
    return dice


def dice_coef_image_level(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    if dim == (2, 3):
        mean_dim = (1, 0)
    elif dim == (1, 2):
        mean_dim = 0
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=1)
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def hd_dist(y_true, y_pred):
    preds_coords = np.argwhere(y_pred.detach().cpu().numpy())
    targets_coords = np.argwhere(y_true.detach().cpu().numpy())
    haussdorf_dist = directed_hausdorff(preds_coords, targets_coords)[0]
    return haussdorf_dist


def criterion(y_pred, y_true):
    loss = 0.0
    for func_name, weight in CFG.loss_info.items():
        loss += (loss_dict[func_name](y_pred, y_true)) * weight
    return loss


# ## Train Utils


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
            gt_label = (masks.sum(dim=(1, 2, 3)) > 0).type("torch.FloatTensor").cuda()
            loss = criterion(
                y_pred, masks
            )  # + nn.BCEWithLogitsLoss()(y_label.squeeze(), gt_label) #+ nn.MSELoss()(reg_out, reg_label)
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
        pbar.set_postfix(train_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_mem=f"{mem:0.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_dict = {"V_Dice": [], "V_Jaccard": [], "V_HD_Dist": [], "V_Dice_LB": [], "V_Dice_SB": [], "V_Dice_ST": []}

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Valid ")
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)
        y_pred = model(images)
        gt_label = (masks.sum(dim=(1, 2, 3)) > 0).type("torch.FloatTensor").cuda()
        loss = criterion(
            y_pred, masks
        )  # + nn.BCEWithLogitsLoss()(y_label.squeeze(), gt_label)  #+ nn.MSELoss()(reg_out, reg_label)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)

        ## Metrics
        val_dict["V_Dice"].append(dice_coef(masks, y_pred).cpu().detach().numpy())
        val_dict["V_Jaccard"].append(iou_coef(masks, y_pred).cpu().detach().numpy())
        # val_dict['V_HD_Dist'].append(hd_dist(masks, y_pred))
        val_dict["V_Dice_LB"].append(
            dice_coef(masks[:, 0, :, :], y_pred[:, 0, :, :], dim=(1, 2)).cpu().detach().numpy()
        )
        val_dict["V_Dice_SB"].append(
            dice_coef(masks[:, 1, :, :], y_pred[:, 1, :, :], dim=(1, 2)).cpu().detach().numpy()
        )
        val_dict["V_Dice_ST"].append(
            dice_coef(masks[:, 2, :, :], y_pred[:, 2, :, :], dim=(1, 2)).cpu().detach().numpy()
        )

        mem = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(valid_loss=f"{epoch_loss:0.4f}", lr=f"{current_lr:0.5f}", gpu_memory=f"{mem:0.2f} GB")
    for k, v in val_dict.items():
        val_dict[k] = np.mean(np.array(v))

    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_dict


def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_valid_loss = np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f"Epoch {epoch}/{num_epochs}", end="")
        train_loss = train_one_epoch(
            model, optimizer, scheduler, dataloader=train_loader, device=CFG.device, epoch=epoch
        )

        val_loss, val_dict = valid_one_epoch(model, valid_loader, device=CFG.device, epoch=epoch)

        history["Train Loss"].append(train_loss)
        history["Valid Loss"].append(val_loss)
        for k, v in val_dict.items():
            history[k].append(v)

        # Log the metrics
        val_dict.update({"LR": scheduler.get_last_lr()[0], "Train Loss": train_loss, "Valid Loss": val_loss})
        wandb.log(val_dict)

        print(f'Valid Dice: {val_dict["V_Dice"]:0.4f} | Valid Jaccard: {val_dict["V_Jaccard"]:0.4f}')

        # deep copy the model
        if val_dict["V_Dice"] >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dict['V_Dice']:0.4f})")
            best_dice = val_dict["V_Dice"]
            best_jaccard = val_dict["V_Jaccard"]
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            # PATH = f"{CFG.MODEL_PATH}best_epoch.bin"
            # torch.save(model.state_dict(), PATH)
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/{CFG.WT_NAME}_model_fold{fold}.bin")
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{OUTPUT_DIR}/{CFG.WT_NAME}_other_fold{fold}.bin",
            )
            # Save a model file from the current directory
            # wandb.save(PATH)
            print(f"Model Saved{sr_}")

        # if val_loss < best_valid_loss:
        #     LS_PATH = f"{CFG.MODEL_PATH}best_loss.bin"
        #     torch.save(model.state_dict(), LS_PATH)

        # last_model_wts = copy.deepcopy(model.state_dict())
        # PATH = f"{OUTPUT_DIR}/last_epoch-{fold:02d}.bin"
        # torch.save(model.state_dict(), PATH)

        print()
        print()

    end = time.time()
    time_elapsed = end - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60
        )
    )
    print("Best Score: {:.4f}".format(best_jaccard))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CFG.scheduler == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, eta_min=CFG.min_lr)
    elif CFG.scheduler == "ReduceLROnPlateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)
    elif CFG.scheduer == "ExponentialLR":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler == None:
        return None

    return scheduler


# ## Model Utils


def change_relu(model, new_act):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, new_act)
        else:
            change_relu(child, new_act)


class FinalActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0)


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


train_loader, valid_loader = prepare_loaders(fold=0, debug=True)
imgs, msks = next(iter(train_loader))
imgs.size(), msks.size()
# plot_batch(imgs, msks, size=5)


# ### Training

for fold in CFG.folds:
    print(f"#" * 15)
    print(f"### Fold: {fold}")
    print(f"#" * 15)
    gc.collect()
    set_seed(CFG.seed)

    run = wandb.init(
        project="UWMGI",
        config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
        name=f"FOLD{fold}_{CFG.WANDB_NAME}",
    )
    gc.collect()
    set_seed(CFG.seed)

    train_loader, valid_loader = prepare_loaders(fold=fold, debug=CFG.debug)
    model = build_model()
    gc.collect()
    set_seed(CFG.seed)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)
    gc.collect()
    set_seed(CFG.seed)

    model, history = run_training(model, optimizer, scheduler, device=CFG.device, num_epochs=CFG.epochs)
    run.finish()
    gc.collect()
    set_seed(CFG.seed)

#     display(ipd.IFrame(run.url, width=1000, height=720))


# ## Prediction

test_dataset = BuildDataset(df.query("fold==0").sample(frac=1.0), label=False, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=5, num_workers=4, shuffle=False, pin_memory=False)
imgs = next(iter(test_loader))
imgs = imgs.to(CFG.device, dtype=torch.float)

preds = []
for fold in range(1):
    model = load_model(f"{OUTPUT_DIR}/{CFG.WT_NAME}_model_fold{fold}.bin")
    with torch.no_grad():
        pred, _ = model(imgs)
        pred = (nn.Sigmoid()(pred) > 0.5).double()
    preds.append(pred)

imgs = imgs.cpu().detach()
preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()

# plot_batch(imgs, preds, size=5)


# ## Test Predictions


test_dataset = BuildDataset(df, label=True, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=48, num_workers=4, shuffle=False, pin_memory=True)


model = load_model(f"{OUTPUT_DIR}/{CFG.WT_NAME}_model_fold{fold}.bin")


score_ls = []
preds_cpu = None
for ii, (imgs, masks) in tqdm(enumerate(test_loader)):
    with torch.no_grad():
        imgs = imgs.to(CFG.device, dtype=torch.float)
        pred, _ = model(imgs)
        pred = (nn.Sigmoid()(pred) > 0.5).double()

        score = dice_coef_image_level(masks.cpu().detach(), pred.cpu().detach())
        score_ls.extend(score)
        score_msk = score < 0.4

        if score_msk.any():
            if preds_cpu == None:
                preds_cpu = pred[score_msk, :, :, :].cpu().detach()
                imgs_cpu = imgs[score_msk, :, :, :].cpu().detach()
                masks_cpu = masks[score_msk, :, :, :].cpu().detach()
            else:
                preds_cpu = torch.cat([preds_cpu, pred[score_msk, :, :, :].cpu().detach()])
                imgs_cpu = torch.cat([imgs_cpu, imgs[score_msk, :, :, :].cpu().detach()])
                masks_cpu = torch.cat([masks_cpu, masks[score_msk, :, :, :].cpu().detach()])


# imgs_cpu.shape, preds_cpu.shape, masks_cpu.shape, len(imgs_cpu)


# plt.hist(score_ls)


# #### Image Level Dice

# score_arr = np.array(score_ls)
# score_arr[(score_arr < 0.4)].shape, score_arr.shape


# for ii in range(len(imgs_cpu)):
#     img = imgs_cpu[ii, :, :, :].permute((1, 2, 0)).numpy() * 10.0
#     print(f"Processing {ii}", np.max(img), np.min(img))
#     msk = masks_cpu[ii, :, :, :].permute((1, 2, 0)).numpy() * 255.0
#     preds = preds_cpu[ii, :, :, :].permute((1, 2, 0)).numpy() * 255.0

#     f, axarr = plt.subplots(1, 3, figsize=(10, 5))
#     axarr[0].imshow(img)
#     axarr[1].imshow(msk)
#     axarr[2].imshow(preds)
#     plt.show()


# (preds_cpu > 0).sum(), (masks_cpu > 0).sum()
