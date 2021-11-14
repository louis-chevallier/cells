






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


#get_ipython().run_cell_magic('capture', '', '!pip install ../input/segmentation-models-wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/pretrainedmodels-0.7.4-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/timm-0.3.2-py3-none-any.whl\n!pip install ../input/segmentation-models-wheels/segmentation_models_pytorch-0.1.3-py3-none-any.whl')


# # Import Dependencies

# In[ ]:

import argparse

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
import albumentations as A
import tqdm
from albumentations.pytorch import ToTensorV2
warnings.filterwarnings("ignore")
import utillc
from utillc import *
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
#import segmentation_models_pytorch as smp
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import torchvision

EKOT(utillc.__file__)


# # Define constants and load df

# In[ ]:
SAMPLE_SUBMISSION  = 'inputs/sample_submission.csv'
TRAIN_CSV = "inputs/train.csv"
TRAIN_PATH = "inputs/train"
TEST_PATH = "inputs/test"

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

# (336, 336)
IMAGE_RESIZE = (224, 224)

LEARNING_RATE = 5e-4



################ a copier coller dans le notebook de kaggle

class Nop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = tuple(args)
        info(args)
    def forward(self, x):
        b = x.shape[0]
        return x.view((b, *self.shape))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Print(nn.Module):
    dump=False #True
    def __init__(self, name=""):
        super().__init__()
        self.name = name
    def forward(self, x):
        if Print.dump :
            info((self.name, x.shape))
        return x

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = torchvision.models.wide_resnet50_2(pretrained=False)
        self.base_model = torchvision.models.resnet18()

        #self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        #self.layer1_1x1 = convrelu(256, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        #self.layer2_1x1 = convrelu(512, 128, 1, 0)

        self.layer3 = self.base_layers[6]

        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        #self.layer3_1x1 = convrelu(1024, 256, 1, 0)
        self.layer4 = self.base_layers[7]

        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        #self.layer4_1x1 = convrelu(2048, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

        self.L = nn.Sequential(
            nn.Linear(25088, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 1))

        #            nn.ReLU(inplace=True))

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        bs, _, _, _ = layer4.shape
        xx = layer4.view(bs, -1)
        nob = self.L(xx)

        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)
        return out, nob



class Mean(nn.Module):
    dump=True
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.mean((2,3))

def subset(dataset, n) :
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:n]
    return torch.utils.data.Subset(dataset, indices)

class Cell :
    def __init__(self, args) :
        self.args = args
        EKOX(args.debug)
        pass
        self.EPOCHS = 12
        self.PARALLEL=True
        self.BATCH_SIZE=4
        self.SUBSET=99999
        self.debug = args.debug
        self.WORKERS=4
        self.tqdmf = tqdm.tqdm

        if self.debug :
            self.EPOCHS=2
            self.SUBSET = 12
            self.WORKERS=0
            self.tqdmf = lambda x, total=-1 : x


    def run(self) :

        prefix = "_" + self.args.model 
        df_train = pd.read_csv(TRAIN_CSV)
        EKOX(df_train.head())

        # # Training Dataset and Dataloader

        # ## Utilities

        # In[ ]:

        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
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
            objs = []
            for label in labels:
                m = rle_decode(label, shape=(height, width))
                mask += m
                objs.append(m)
            mask = mask.clip(0, 1)
            #EKOX(len(labels))
            return mask, len(labels), objs


        # ## Dataset and Dataloader

        # In[ ]:


        class CellDataset(Dataset):
            def __init__(self, df, withObjects = False, lst = None):
                self.df = df
                self.base_path = TRAIN_PATH
                self.transforms = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), 
                                           Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1),  
                                           A.Affine(scale={ "x" : (0.9, 1.1), "y" :  (0.8, 1.2)}, 
                                                    translate_percent=(-0.1, 0.1), 
                                                    rotate=(50, -50), 
                                                    shear=(10, -10), mode=cv2.BORDER_REFLECT ),

                                           ToTensorV2()])
                self.gb = self.df.groupby('id')
                self.image_ids = df.id.unique().tolist()
                if lst is not None :self.image_ids = lst                    
                EKOX(TYPE(self.image_ids))
                random.shuffle(self.image_ids)
                self.withObjects = withObjects
            def __getitem__(self, idx):
                image_id = self.image_ids[idx]
                df = self.gb.get_group(image_id)
                annotations = df['annotation'].tolist()
                image_path = os.path.join(self.base_path, image_id + ".png")
                image = cv2.imread(image_path)
                mask, nobjects, objects = build_masks(df_train, image_id, input_shape=(520, 704))
                mask = (mask >= 1).astype('float32')
                objects = (np.asarray(objects) > 1).astype('float32')

                if self.withObjects :
                    augmented = self.transforms(image=image, mask=mask, masks=list(objects))
                    image = augmented['image']
                    mask = augmented['mask']
                    masks = augmented['masks']
                    l1 = [ masks[0] ] * 800
                    masks = np.asarray(masks)
                    masks1 = np.asarray(l1)
                    masks1[0:nobjects, : , :] = masks
                    EKOX(TYPE(masks1))

                    return image, mask.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1])), nobjects, masks1
                else :
                    augmented = self.transforms(image=image, mask=mask)
                    image = augmented['image']
                    mask = augmented['mask']
                    return image, mask.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1])), nobjects
                    

            def __len__(self):
                return len(self.image_ids)


        # In[ ]:


        ds_train_0 = CellDataset(df_train)
        nimage = len(ds_train_0.image_ids)
        ds_train = CellDataset(df_train, lst=ds_train_0.image_ids[nimage * 4 // 5 : ])
        ds_val = CellDataset(df_train,  lst=ds_train_0.image_ids[: nimage // 5], withObjects = True)
        image, mask, nobjs = ds_train[1]
        EKOX((image.shape, mask.shape))


        # In[ ]:

        """
        plt.imshow(image[0], cmap='bone')
        plt.show()
        plt.imshow(mask[0], alpha=0.3)
        plt.show()
        """

        # In[ ]:

        class Model(nn.Module) :
            def __init__(self) :
                super().__init__()
                self.segnet = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, aux_loss=False)
                self.classifier = DeepLabHead(2048, num_classes=1)
                self.segnet.classifier = Nop()


            def forward(self, input):
                y = self.segnet(input)['out']
                y = self.classifier(y)
                y = torch.sigmoid(y)
                return y

                
        model = {
            "deeplab" : lambda : Model(),
            "unet" : lambda : UNet(1)
        }[self.args.model]()
        #model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)

        model.cuda()
        model.train()


        if self.PARALLEL :
            model = nn.DataParallel(model)


        #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        optimizer = torch.optim.Adam([ { 'params' : model.parameters(), 'lr':1e-4 }])

        info("loading cpts")
        try :
            model.load_state_dict(torch.load("model_%s.cpt" % prefix))
            info("cpt loaded")
        except Exception as e :
            info("not loaded")

        ious = []

        EKOT("stats ..")
        # a faire une fois
        # 800 objects max
        if False :
            dl_train = DataLoader(dataset=ds_train, batch_size=self.BATCH_SIZE, num_workers=self.WORKERS, shuffle=False) # pin_memory=True, 
            lobjs = [ n for batch_idx, batch in enumerate(dl_train) for n in batch[2]]
            lobjs = np.asarray(lobjs)
            
            meanNobjs = np.mean(lobjs)
            stdNobjs = np.std(lobjs)
            
            n, bins, patches = plt.hist(lobjs, 128, facecolor='blue', alpha=0.5); plt.show()
            n, bins, patches = plt.hist(np.log(lobjs), 128, facecolor='blue', alpha=0.5); plt.show()
            n, bins, patches = plt.hist(np.log(lobjs) / 6, 128, facecolor='blue', alpha=0.5); plt.show()
        else :
            meanNobjs, stdNobjs = 121., 151.

        EKOX((meanNobjs, stdNobjs))

        for era in range(10) :
            train_len = len(ds_train)
            TS = train_len * 4 // 5
            #train_set, val_set = torch.utils.data.random_split(ds_train, [TS, train_len - TS])
            train_set, val_set = ds_train, ds_val

            dl_train = DataLoader(dataset=subset(train_set, self.SUBSET), batch_size=self.BATCH_SIZE, num_workers=self.WORKERS, shuffle=False) # pin_memory=True, 
            dl_val = DataLoader(dataset=subset(val_set, self.SUBSET), batch_size=self.BATCH_SIZE, num_workers=self.WORKERS, shuffle=False) # pin_memory=True, 
            EKOX(len(dl_train))
            n_batches = len(dl_train)
            # get a batch from the dataloader
            batch = next(iter(dl_val))
            batch = next(iter(dl_train))
            images, masks, nobjs = batch

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

            criterion = MixedLoss(10.0, 2.0)                    
            #criterion = torch.nn.CrossEntropyLoss()

            if self.args.model == "deeplab" :
                criterion = torch.nn.BCELoss()

            model.train()
            losses = []
            for epoch in range(1, self.EPOCHS + 1):
                EKOT(f"Starting epoch: {epoch=} / {self.EPOCHS=}")
                running_loss = 0.0
                optimizer.zero_grad()

                for batch_idx, batch in self.tqdmf(enumerate(dl_train), total=len(dl_train)):
                    # Predict
                    images, masks, nobjs = batch
                    images, masks, nobjs = images.cuda(),  masks.cuda(), nobjs.cuda().float()


                    #nobjs = nobjs.log() / 6 # pour rammener ce nombre entre 0 et 1
                    nobjs = (nobjs - meanNobjs) / stdNobjs
                    masks = masks # .round().long()
                    try :
                        outputs, pnob = model(images)
                        #pnob = torch.clamp(pnob, 0, 10).T
                        pnob = pnob.T
                        outputs = outputs.view(self.BATCH_SIZE, -1)
                        masks = masks.view(self.BATCH_SIZE, -1)
                        loss1 = criterion(outputs, masks)
                        loss2 = nn.functional.l1_loss(pnob, nobjs)
                        
                        #EKOX(loss1.item())
                        #EKOX(loss2.item())
                        
                        loss = loss1 + loss2
                        # Back prop
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        running_loss += loss.item()
                    except Exception as e  :
                        EKOX(e)
                        if self.debug : raise(e)
                        pass
                epoch_loss = running_loss / n_batches
                EKOT(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f}")
            EKOT("saving")
            torch.save(model.state_dict(), "model_%s.cpt" % prefix)
            EKO()
            
            model.eval()

            def getimage(x, nb=0) :

                if isinstance(x, torch.Tensor) : x = x.cpu().detach().numpy()

                x = x[nb,0]
                x = np.dstack((x,x,x)) * 255
                x = x.astype(int)
                return x


            # ## Postprocessing: separate different components of the prediction mask

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
                return predictions, num_component

            def cnob(probability_mask) :
                try:
                    EKOX(TYPE(probability_mask))
                    #if probability_mask.shape != IMAGE_RESIZE:
                    #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                    probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
                    objs, nobjs = post_process(probability_mask)
                    #EKOX(nobjs)
                except Exception as e:
                    info(e)
                    pass
                return nobjs, objs

            def match(pred, mask, th) :
                EKOX(TYPE(pred))
                EKOX(TYPE(mask))
                mask = cv2.resize(mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
                pred = (pred > th).astype(int)
                inter = pred * mask
                union = pred + mask - inter
                iou = inter.sum() / (union.sum() + 1)
                return iou > th

            def score(preds, gts, th) :
                # par image : preds = mask of objs, gts=objs
                EKOX(preds.shape)
                EKOX(gts.shape)
                EKOX(TYPE(preds))
                def c1(od, gts) :
                    EKOX(TYPE(od))
                    EKOX(TYPE(gts))
                    mm = [ match(od, ogt, th) for ogt in gts]
                    tp = sum(mm) > 0
                    fp = sum(mm) == 0
                    return tp, fp

                opreds = cnob(preds)[1]
                EKOX(TYPE(opreds))

                l = [ c1(od, gts) for od in opreds ]
                TP = sum([ el[0] for el in l]) 
                FP = sum([ el[1] for el in l]) 
                EKOX((TP, FP))
                l = [ c1(od, opreds) for od in gts ]
                FN = sum([ el[1] for el in l])
                EKOX(FN)
                return TP, FP, FN


            def fbatch(i, batch, th=0.5) :
                images, masks, nobjs, maskss = batch
                preds, pnobjs = model(images.cuda())
                preds = torch.sigmoid(preds)
                EKOX(pnobjs.T.detach().cpu().numpy())                
                #pnobjs = (pnobjs * 6).exp() # inversion de la normalization
                pnobjs = (pnobjs * stdNobjs) + meanNobjs

                predsP = preds.detach().cpu().numpy() # (batch_size, 1, size, size) -> (batch_size, size, size)
                """
                dd = preds.flatten()
                n, bins, patches = plt.hist(dd, 128, facecolor='blue', alpha=0.5); plt.show()
                dd = masks.cpu().numpy().flatten()
                n, bins, patches = plt.hist(dd, 128, facecolor='blue', alpha=0.5); plt.show()
                """
                masks = masks.cpu().numpy()

                EKOX(maskss.shape)

                preds = (predsP > th).astype(int)
                #masks = preds
                #EKOX(preds.sum())
                #EKOX(masks.sum())

                inter = preds * masks
                union = preds + masks - inter
                iou = inter.sum() / (union.sum() + 1)
                ii = np.hstack((
                    (images[0].permute((1,2,0)).detach().cpu().numpy() * 255).astype(int),
                    getimage(preds),
                    getimage(masks),
                    getimage(inter),
                    getimage(union)))

                if i == 0 : 
                    EKOI(ii)
                    EKOX(iou)

                predsQ = predsP[:, 0, :, :]
                EKOX([ cnob(probability_mask)[0] for probability_mask in predsQ])
                EKOX(pnobjs.T.detach().cpu().numpy())
                EKOX([ cnob(probability_mask)[0] for probability_mask in masks[:,0,:,:]])
                EKOX(nobjs.T)
                EKOX(masks)
                masksT = masks[:,0,:,:]
                EKOX(TYPE(cnob(predsQ[0])[1]))
                EKOX(TYPE(cnob(masksT[0])[1]))

                maskss = maskss.cpu().numpy()
                lscore = [ score(preds, gts, th) for preds, gts in zip(predsQ, maskss)]
                EKOX(lscore)



                EKOX(predsQ.shape)
                EKOX(masks.shape)




                if self.debug :
                    EKOX(TYPE(preds))
                    EKOI(ii)
                    EKOX(pnobjs)
                    """
                    EKOI((images[0].permute((1,2,0)).detach().cpu().numpy() * 255).astype(int))
                    EKOI(getimage(inter))
                    EKOI(getimage(union))
                    EKOI(getimage(preds))
                    EKOI(getimage(masks))
                    """
                    EKOX(iou)
                    #EKOX(preds.shape)
                    #EKOX(masks.shape)
                return iou
            EKOX(np.mean([ fbatch(i, batch) for i, batch in enumerate(self.tqdmf(dl_val))]))

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


            ds_test = TestCellDataset()
            dl_test = DataLoader(ds_test, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


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

            # I am not sure if they adapt the sample submission csv or only the test folder
            # I am using the test folders as the ground truth for the images to predict, which should be always right
            # The sample csv is ignored
            pd.read_csv(SAMPLE_SUBMISSION)

            # ### Predict & submit

            model.eval()

            submission = []
            for i, batch in enumerate(self.tqdmf(dl_test)):
                preds, pnobjs = model(batch['image'].cuda())

                #pnobjs = (pnobjs * 6).exp() # inversion de la normalization
                pnobjs = (pnobjs * stdNobjs) + meanNobjs
                
                preds = torch.sigmoid(preds)
                preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
                for image_id, probability_mask in zip(batch['id'], preds):
                    try:
                        #if probability_mask.shape != IMAGE_RESIZE:
                        #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                        probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
                        predictions, nobjs = post_process(probability_mask)
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
                        if self.debug : raise(e)

            df_submission = pd.DataFrame(submission, columns=['id', 'predicted'])
            df_submission.to_csv('submission_%s_%04d.csv' % (prefix, era), columns=['id', 'predicted'], index=False)
            df_submission.to_csv('submission.csv', columns=['id', 'predicted'], index=False)

            EKOX(df_submission.head())


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, default="")
    parser.add_argument('--debug', type=Bool, default=False)
    parser.add_argument('--model', default="deeplab")
    args = parser.parse_known_args()[0]
    
    fs = Cell(args)
    fs.run()
