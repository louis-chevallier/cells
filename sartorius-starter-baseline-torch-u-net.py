#!/usr/bin/env python
# coding: utf-8

# # Baseline Torch U-net with Resnet34
# 
# This is a very fast implementation of a U-net with a Resnet34, using torch. Both train an inference happen in this notebook.
# 
# The code is simple enough to be the starting building block of a more robust system, but I wanted it to cover all the relevant steps at least to a certain extent.
# It correctly articulates the train dataset, trains an U-net for some epochs, obtains results correctly and post-process them to get a valid submission.
# 
# There is no validation and no IoU measurements, and also the resizing is done in a very rough manner. There are various low-hanging fruits with data augmentation too. The post-processing (the splitting of the segmentation mask into different "individual" predictions) is done with conected components.
# 
# This is a full pipeline from zero to a submission with a score greater than zero though, with all the internet dependencies removed, and a lot of places for improvements and quick-wins. 
# 
# 
# The code is mostly an adaption from [this](https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch/notebook) with various specificities for this competition.
# 
# ### DO upvote!!
# 
# 
# ## Resources
# 
# * [UNet with ResNet34 encoder (Pytorch)](https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch/notebook)
# The code is an adaption of this notebook
# 
# * [pytorch-pretrained-image-models](https://www.kaggle.com/bminixhofer/pytorch-pretrained-image-models)
# The "offline" pytorch's resnet weights come from this dataset
# * [segmentation-models-wheels](https://www.kaggle.com/arunmohan003/segmentation-models-wheels)
# The "offline" segmentation library comes from here
# * [hubmap-hacking-the-kidney](https://www.kaggle.com/c7934597/hubmap-hacking-the-kidney)
# I learned how to use the the previous library from this notebook
# 
# 
# 
# 

# # Install libraries (offline)

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install ../input/segmentation-models-wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/pretrainedmodels-0.7.4-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/timm-0.3.2-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/segmentation_models_pytorch-0.1.3-py3-none-any.whl')


# # Import Dependencies

# In[ ]:


import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")


# # Define constants and load df

# In[ ]:


SAMPLE_SUBMISSION  = '../input/sartorius-cell-instance-segmentation/sample_submission.csv'
TRAIN_CSV = "../input/sartorius-cell-instance-segmentation/train.csv"
TRAIN_PATH = "../input/sartorius-cell-instance-segmentation/train"
TEST_PATH = "../input/sartorius-cell-instance-segmentation/test"

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

# (336, 336)
IMAGE_RESIZE = (224, 224)

LEARNING_RATE = 5e-4
EPOCHS = 12


# In[ ]:


df_train = pd.read_csv(TRAIN_CSV)
df_train.head()


# # Training Dataset and Dataloader

# ## Utilities

# In[ ]:


def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)


def build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return mask


# ## Dataset and Dataloader

# In[ ]:


class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.base_path = TRAIN_PATH
        self.transforms = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1), ToTensorV2()])
        self.gb = self.df.groupby('id')
        self.image_ids = df.id.unique().tolist()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)
        mask = build_masks(df_train, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)


# In[ ]:


ds_train = CellDataset(df_train)
image, mask = ds_train[1]
image.shape, mask.shape


# In[ ]:


plt.imshow(image[0], cmap='bone')
plt.show()
plt.imshow(mask[0], alpha=0.3)
plt.show()


# In[ ]:


dl_train = DataLoader(ds_train, batch_size=8, num_workers=4, pin_memory=True, shuffle=False)


# In[ ]:


len(dl_train)


# In[ ]:


# get a batch from the dataloader
batch = next(iter(dl_train))
images, masks = batch


# In[ ]:


idx=1
plt.imshow(images[idx][0], cmap='bone')
plt.show()
plt.imshow(masks[idx][0], alpha=0.3)
plt.show()
plt.imshow(images[idx][0], cmap='bone')
plt.imshow(masks[idx][0], alpha=0.3)
plt.show()


# # Losses

# In[ ]:


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val +             ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()


# # Model: U-net
# 
# In order to comply with the no-Internet restriction I am placing the resnet from a dataset into the `.cache` path.
# Also, there is an error when importing `segmentation_models_pytorch`, that I solved setting a private attributre in `torch` 🤦‍♂️.
# 
# I solved both problems in a quick-and-dirty way for now.

# In[ ]:


get_ipython().system('mkdir -p /root/.cache/torch/hub/checkpoints/')
get_ipython().system('cp ../input/pytorch-pretrained-image-models/resnet34.pth /root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth')

import torch
import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp


# # U-Net

# In[ ]:


model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)


# In[ ]:


# Check model details
# model


# # Training loop
# 
# No validation or k-folds for now, just get it running for few epochs.

# In[ ]:


torch.set_default_tensor_type("torch.cuda.FloatTensor")
n_batches = len(dl_train)

model.cuda()
model.train()

criterion = MixedLoss(10.0, 2.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS + 1):
    print(f"Starting epoch: {epoch} / {EPOCHS}")
    running_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dl_train):
        
        # Predict
        images, masks = batch
        images, masks = images.cuda(),  masks.cuda()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Back prop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    epoch_loss = running_loss / n_batches
    print(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f}")


# # Predict

# ## Test Dataset and DataLoader

# In[ ]:


class TestCellDataset(Dataset):
    def __init__(self):
        self.test_path = TEST_PATH
        
        # I am not sure if they adapt the sample submission csv or only the test folder
        # I am using the test folders as the ground truth for the images to predict, which should be always right
        # The sample csv is ignored
        self.image_ids = [f[:-4]for f in os.listdir(self.test_path)]
        self.num_samples = len(self.image_ids)
        self.transform = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1), ToTensorV2()])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_path, image_id + ".png")
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        return {'image': image, 'id': image_id}

    def __len__(self):
        return self.num_samples


# In[ ]:


del dl_train, ds_train, optimizer


# In[ ]:


ds_test = TestCellDataset()
dl_test = DataLoader(ds_test, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)


# ## Postprocessing: separate different components of the prediction mask

# ### Utilities

# In[ ]:


def post_process(probability, threshold=0.5, min_size=300):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = []
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            a_prediction = np.zeros((520, 704), np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions

# Stolen from: https://www.kaggle.com/arunamenon/cell-instance-segmentation-unet-eda
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
# Modified by me
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))


# In[ ]:


# I am not sure if they adapt the sample submission csv or only the test folder
# I am using the test folders as the ground truth for the images to predict, which should be always right
# The sample csv is ignored
pd.read_csv(SAMPLE_SUBMISSION)


# ### Predict & submit

# In[ ]:


model.eval()

submission = []
for i, batch in enumerate(tqdm(dl_test)):
    preds = torch.sigmoid(model(batch['image'].cuda()))
    preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
    for image_id, probability_mask in zip(batch['id'], preds):
        try:
            #if probability_mask.shape != IMAGE_RESIZE:
            #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
            probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
            predictions = post_process(probability_mask)
            for prediction in predictions:
                #plt.imshow(prediction)
                #plt.show()
                try:
                    submission.append((image_id, rle_encoding(prediction)))
                except:
                    print("Error in RL encoding")
        except Exception as e:
            print(f"Exception for img: {image_id}: {e}")
            submission.append((image_id, ""))
            
df_submission = pd.DataFrame(submission, columns=['id', 'predicted'])
df_submission.to_csv('submission.csv', columns=['id', 'predicted'], index=False)


# In[ ]:


df_submission.head()


# # _Please DO upvote_
