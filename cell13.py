"""

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:04:07.470213Z","iopub.execute_input":"2021-11-07T00:04:07.470753Z","iopub.status.idle":"2021-11-07T00:05:33.973040Z","shell.execute_reply.started":"2021-11-07T00:04:07.470707Z","shell.execute_reply":"2021-11-07T00:05:33.972020Z"}}
#%%capture
#!pip install ../input/segmentation-models-wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl
#!pip install ../input/segmentation-models-wheels/pretrainedmodels-0.7.4-py3-none-any.whl
#!pip install ../input/segmentation-models-wheels/timm-0.3.2-py3-none-any.whl
#!pip install ../input/segmentation-models-wheels/segmentation_models_pytorch-0.1.3-py3-none-any.whl

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T09:51:46.162327Z","iopub.execute_input":"2021-11-07T09:51:46.162618Z","iopub.status.idle":"2021-11-07T09:51:48.153785Z","shell.execute_reply.started":"2021-11-07T09:51:46.162584Z","shell.execute_reply":"2021-11-07T09:51:48.152928Z"}}
!pwd
!ls
!ls ../input/sartorius-cell-instance-segmentation/train
"""


"""
a partir d'une image carte contenent N segments ( N 400 )
 coloriee avec N couleurs
la fct F produit une carte utilisant un nombre fixe m de couleurs  ( normalement 4 suffisent ) 

on demande a F de produire un coloriage 'stable' : des changements locaux de la forme / nombre de segment, n'affecte pas radicalement l'affectation 
des couleurs 

loss = nombre total de  couleurs employées + nombre de segment voisins ayant une couleur identique


"""


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:41.507344Z","iopub.execute_input":"2021-11-07T00:05:41.507924Z","iopub.status.idle":"2021-11-07T00:05:45.343139Z","shell.execute_reply.started":"2021-11-07T00:05:41.507879Z","shell.execute_reply":"2021-11-07T00:05:45.342381Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print("hello")
from functools import partial
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
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensorV2
#import kornia
#import canny
from torch import autograd
warnings.filterwarnings("ignore")


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)

"""
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:45.344708Z","iopub.execute_input":"2021-11-07T00:05:45.344996Z","iopub.status.idle":"2021-11-07T00:05:49.322138Z","shell.execute_reply.started":"2021-11-07T00:05:45.344959Z","shell.execute_reply":"2021-11-07T00:05:49.321156Z"}}
!pwd
!ls
!ls ../input/sartorius-cell-instance-segmentation
!cd ..
!pwd
!mkdir /kaggle/working/images
"""

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:49.324361Z","iopub.execute_input":"2021-11-07T00:05:49.324668Z","iopub.status.idle":"2021-11-07T00:05:49.523974Z","shell.execute_reply.started":"2021-11-07T00:05:49.324632Z","shell.execute_reply":"2021-11-07T00:05:49.523292Z"}}
import utillc
from utillc import *
a=1
utillc._el, utillc._readEL = 0, False
EKOX(a)
EKOX(utillc.tempDir)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:50.537071Z","iopub.execute_input":"2021-11-07T00:05:50.537887Z","iopub.status.idle":"2021-11-07T00:05:51.909357Z","shell.execute_reply.started":"2021-11-07T00:05:50.537836Z","shell.execute_reply":"2021-11-07T00:05:51.908679Z"}}
#dir(utillc)
#print(utillc.__file__)
#print(dir(utillc))
import scipy
import scipy.misc
import matplotlib.pyplot as plt

f = scipy.misc.face()
#plt.imshow(f)
EKOI(f)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:51.910661Z","iopub.execute_input":"2021-11-07T00:05:51.910910Z","iopub.status.idle":"2021-11-07T00:05:51.916757Z","shell.execute_reply.started":"2021-11-07T00:05:51.910879Z","shell.execute_reply":"2021-11-07T00:05:51.915868Z"}}
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


input = os.path.join('..', 'input') 
input = './inputs' if not os.path.exists(input) else input

input = os.path.join(input, 'sartorius-cell-instance-segmentation')

SAMPLE_SUBMISSION  = os.path.join(input, 'sample_submission.csv')
TRAIN_CSV = os.path.join(input, "train.csv")
TRAIN_PATH = os.path.join(input, "train")
TEST_PATH = os.path.join(input, "test")

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)

# (336, 336)
IMAGE_RESIZE = (224*2, 224*2)
#IMAGE_RESIZE = (224, 224)

DSIZE = (224, 224)
DSIZE = (520, 704)
BORDER= 5
kernel = np.ones((2, 2), np.uint8) 
SWEEP = np.arange(0.5, 0.95, 0.05)
rev = lambda x : tuple(list(x)[::-1])
LEARNING_RATE = 5e-4

max_objs = 500
colors = np.random.uniform(0, 255, (max_objs, 3))

#cannyf = kornia.filters.Canny(low_threshold=0.4, high_threshold=0.99)


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:52.068697Z","iopub.execute_input":"2021-11-07T00:05:52.069074Z","iopub.status.idle":"2021-11-07T00:05:52.092206Z","shell.execute_reply.started":"2021-11-07T00:05:52.069033Z","shell.execute_reply":"2021-11-07T00:05:52.091509Z"}}

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

import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn

import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
#import segmentation_models_pytorch as smp
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import torchvision
import logging
logging.basicConfig(level=logging.INFO, format='%(pathname)s:%(lineno)d: [%(asctime)ss%(msecs)03d]:%(message)s', datefmt='%Hh%Mm%S')
logging.info("")
from logging import info

'''
def EKO() : print("")
def EKOX(x) : print(x)
def EKOI(x) : pass
def EKOT(x) : print(x)

'''

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:52.697409Z","iopub.execute_input":"2021-11-07T00:05:52.698050Z","iopub.status.idle":"2021-11-07T00:05:52.707394Z","shell.execute_reply.started":"2021-11-07T00:05:52.698011Z","shell.execute_reply":"2021-11-07T00:05:52.706580Z"}}

class Nop(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = tuple(args)
        EKOX(args)
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
            EKOX((self.name, x.shape))
        return x

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:53.443050Z","iopub.execute_input":"2021-11-07T00:05:53.443314Z","iopub.status.idle":"2021-11-07T00:05:53.464742Z","shell.execute_reply.started":"2021-11-07T00:05:53.443286Z","shell.execute_reply":"2021-11-07T00:05:53.464061Z"}}

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
        DD = {
            (224*2, 224*2) :  100352,
            (224, 224) : 25088
        }[IMAGE_RESIZE]
        self.L = nn.Sequential(
            nn.Linear(DD, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 1))
        self.LT = nn.Sequential(
            nn.Linear(DD, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 3))


        #            nn.ReLU(inplace=True))

    def forward(self, input):
        xx = x_original = self.conv_original_size0(input)
        xx = x_original = self.conv_original_size1(x_original)
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        xx = layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)
        bs, _, _, _ = layer4.shape
        xx = layer4.view(bs, -1)
        nob = self.L(xx)
        typ = self.LT(xx)

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
        return out, nob, typ

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:53.984750Z","iopub.execute_input":"2021-11-07T00:05:53.985231Z","iopub.status.idle":"2021-11-07T00:05:53.991217Z","shell.execute_reply.started":"2021-11-07T00:05:53.985187Z","shell.execute_reply":"2021-11-07T00:05:53.990174Z"}}


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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:05:54.728938Z","iopub.execute_input":"2021-11-07T00:05:54.729821Z","iopub.status.idle":"2021-11-07T00:05:55.302046Z","shell.execute_reply.started":"2021-11-07T00:05:54.729781Z","shell.execute_reply":"2021-11-07T00:05:55.301323Z"}}
df_train = pd.read_csv(TRAIN_CSV)
#EKOX(df_train.head())
l = list(df_train.cell_type)
df_train.cell_type.unique().tolist()

cell_types = df_train.cell_type.unique().tolist()

cell_types.index('cort')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:21:20.411101Z","iopub.execute_input":"2021-11-07T00:21:20.411876Z","iopub.status.idle":"2021-11-07T00:21:20.501134Z","shell.execute_reply.started":"2021-11-07T00:21:20.411808Z","shell.execute_reply":"2021-11-07T00:21:20.500320Z"}}

class Cell :
    def __init__(self, args) :
        self.args = args
        EKOX(args.debug)
        pass
        self.EPOCHS = 12
        self.PARALLEL=True
        self.BATCH_SIZE=8
        self.SUBSET=99999
        self.debug = args.debug
        self.WORKERS=8
        self.tqdmf = tqdm.tqdm
        self.tqdmf = lambda x, total=-1 : x
        self.ERAS=100
        self.EPOCHS = 10

        if self.debug :
            self.PARALLEL=True #False
            self.ERAS=1
            self.BATCH_SIZE=2
            self.EPOCHS=1
            self.SUBSET = 4
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


        def extractMasks(mask, dsize=DSIZE, min_size=300 ) :
            lp = set(list(mask.flatten()))
            predictions = []  
            if 0 in lp :
                lp.remove(0)
                if rev(dsize) != mask.shape : 
                    mask = cv2.resize(mask, dsize=dsize, interpolation=cv2.INTER_LINEAR)                    
                    #EKOX(probability.shape)

                for i in lp :
                    mm = (mask == i)
                    if mm.astype(int).sum() > min_size:
                        prediction = np.zeros(rev(dsize), np.float32)
                        prediction[mm] = 1
                        predictions.append(prediction)
            return predictions, len(lp)

        def post_process(probability, threshold=0.5, min_size=300, dsize=DSIZE):
            #EKOX(dsize)
            #EKOX(probability.shape)
            if rev(dsize) != probability.shape : 
                #EKO()
                probability = cv2.resize(probability, dsize=dsize, interpolation=cv2.INTER_LINEAR)                    
                #EKOX(probability.shape)

            mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1] 
            #EKOX(probability.shape)
            #EKOX(TYPE(mask))
            mask = mask.round()
            num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
            predictions = []

            for c in range(1, num_component):
                p = (component == c)
                if p.sum() > min_size:
                    a_prediction = np.zeros(rev(dsize), np.float32)
                    a_prediction[p] = 1
                    predictions.append(a_prediction)

            return predictions, num_component

        def erode(mask) :
            image = cv2.erode(mask, kernel)  
            return image

        def build_masks(df_train, image_id, input_shape, max_weight = 1000000000) :
            height, width = input_shape
            labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
            mask = np.zeros((height, width))
            maskC = np.zeros((height, width))
            maskColors = np.zeros((height, width, 3))
            for i, label in enumerate(labels) :
                emask1 = rle_decode(label, shape=(height, width), color=1)
                if True : #emask1.sum() < max_weight :
                    emask = emask1 * (i+1)
                    emask[mask != 0] = 0
                    maskC += emask
                    
                    maskColors += np.dstack((emask1, emask1, emask1)) * colors[i % max_objs]

                    #EKOX(emask1.sum())
                    """
                    for e in range(BORDER) : 
                        mask += emask1 / BORDER
                        emask1 = erode(emask1)
                    """
                    mask += emask1
            """
            #maskC = np.zeros((height, width))
            K = 20
            def patch(x, y, c) :   maskC[y*K : (y+1)*K, x*K : (x+1)*K] = c 
            patch(0, 0, 1.)
            patch(0, 1, 2.)
            patch(1, 1, 4.)
            patch(1, 0, 8.)
            patch(2, 0, 16.)
            patch(3, 0, 32.)
            """

            #EKOX(TYPE(maskC))
            """
            data: torch.tensor = kornia.image_to_tensor(maskC, keepdim=False)  # BxCxHxW
            
            x_magnitude, x_canny = cannyf(data.float())
            x_magnitude, x_canny = canny.canny(data.float())
            #img_canny: np.ndarray = kornia.tensor_to_image(x_canny.byte())
            img_mag: np.ndarray = kornia.tensor_to_image(x_magnitude.byte())
            #n, bins, patches = plt.hist(img_mag.flatten(), 1000); plt.show()
            plt.imshow(img_mag); plt.show()
            plt.imshow(img_canny); plt.show()

            img_mag = img_mag.astype(float)
            img_mag = np.maximum(img_mag, 5.) / 5. 
            img_mag = 1. - img_mag
            img_mag = img_mag * 255

            #EKOX(TYPE(img_canny))
            EKOI(np.dstack((img_canny,img_canny,img_canny)) * 255)
            EKOI(np.dstack((img_mag,img_mag,img_mag)).astype(int))
            """
            mask = mask.clip(0, BORDER)
            return mask, len(labels), maskC, maskColors


        # ## Dataset and Dataloader

        # In[ ]:


        class CellDataset(Dataset):
            def __init__(self, df):
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
                self.cell_types = df.cell_type.unique().tolist()
                self.ltyp = df.cell_type.tolist()
                EKOX(len(self.ltyp))
                EKOX(self.cell_types)
                EKOX(len(self.image_ids))
            def __getitem__(self, idx):
                image_id = self.image_ids[idx]
                df = self.gb.get_group(image_id)
                annotations = df['annotation'].tolist()
                image_path = os.path.join(self.base_path, image_id + ".png")
                image = cv2.imread(image_path)
                hh, ww, _ = image.shape
                assert( (hh, ww) == DSIZE)
                mask, nobjects, maskC, colored = build_masks(df_train, image_id, input_shape=DSIZE)
                """
                EKOX(TYPE(image))
                EKOI(colored.astype(int))
                EKOI(image.astype(int))
                """
                #ppp, nnn = extractMasks(maskC, min_size = 1)
                #EKOX((nobjects, nnn))
                #assert(nnn == nobjects)


                mask = (mask >= 1).astype('float32')
                maskC = maskC.astype('float32')
                augmented = self.transforms(image=image, masks=[mask, maskC])
                image = augmented['image']
                mask, maskC = augmented['masks']
                typ = self.ltyp[idx]
                typ = self.cell_types.index(typ)

                mask = mask.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1]))
                maskC = maskC.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1]))
                #cannyi = cannyi.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1])).astype(float)

                return (image, mask, nobjects, typ, maskC)

            def __len__(self):
                return len(self.image_ids)


        # In[ ]:

        NClasses = 4
       
        EKO()
        model = {
            "deeplab" : lambda : Model(),
            "unet" : lambda : UNet(NClasses)
        }[self.args.model]()
        

        ds_train = CellDataset(df_train)
        image, mask, nobjs, typ, maskC = ds_train[1]
        EKOX((image.shape, mask.shape, typ))


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

        EKO()
        #model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)


        model.cuda()
        model.train()

        EKOX(self.PARALLEL)
        if self.PARALLEL :
            EKO()
            model = nn.DataParallel(model)

        EKOX(count_parameters(model))

        #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        optimizer = torch.optim.Adam([ { 'params' : model.parameters(), 'lr':1e-5 }])

        EKOX("loading cpts")
        try :
            model.load_state_dict(torch.load(self.args.pts))
            EKOT("cpt loaded successfully")
        except Exception as e :
            #EKOX(e)
            EKOX("not loaded")

        ious = []

        EKOT("stats ..")

        dl_train = DataLoader(dataset=ds_train, batch_size=self.BATCH_SIZE, num_workers=12, shuffle=False) # pin_memory=True, 
        for batch in dl_train : 
            model(batch[0].cuda())
            break
        EKO()
        # a faire une fois
        if False :
            dl_train = DataLoader(dataset=ds_train, batch_size=self.BATCH_SIZE, num_workers=12, shuffle=False) # pin_memory=True, 


            l11 = [ mmm.sum() for batch_idx, batch in tqdm.tqdm(enumerate(dl_train), total=len(dl_train)) for maskC in batch[4]  for mmm in extractMasks(maskC[0], dsize=DSIZE, min_size=1)[0]]
            lobjs = [ n for batch_idx, batch in tqdm.tqdm(enumerate(dl_train), total=len(dl_train)) for n in batch[2]]
            lobjs = np.asarray(lobjs)
            
            meanNobjs = np.mean(lobjs)
            stdNobjs = np.std(lobjs)
            
            EKOX(np.sum([e for e in lobjs if e < 100]))
            EKOX(np.sum(lobjs))

            n, bins, patches = plt.hist(l11, 128, facecolor='blue', alpha=0.5); plt.show()

            n, bins, patches = plt.hist(lobjs, 128, facecolor='blue', alpha=0.5); plt.show()
            #n, bins, patches = plt.hist(np.log(lobjs), 128, facecolor='blue', alpha=0.5); plt.show()
            #n, bins, patches = plt.hist(np.log(lobjs) / 6, 128, facecolor='blue', alpha=0.5); plt.show()
        else :
            meanNobjs, stdNobjs = 121., 151.

        EKOX((meanNobjs, stdNobjs))
        train_len = len(ds_train)
        TS = train_len * 4 // 5
        train_set, val_set = torch.utils.data.random_split(ds_train, [TS, train_len - TS])                
        dl_train = DataLoader(dataset=subset(train_set, self.SUBSET), batch_size=self.BATCH_SIZE, num_workers=self.WORKERS, shuffle=False) # pin_memory=True, 
        if False :

            ne = 0
            r = []
            nos = 0
            for batch in self.tqdmf(dl_train) :
                images, masks, nobjs, typ, maskC = batch
                maskC = maskC.cpu().numpy()
                for j in range(len(images)) :
                    gtmasks,  _ = extractMasks(maskC[j, 0], dsize=DSIZE, min_size=0)
                    #EKOX((nobjs[j].int(), len(gtmasks)))
                    #assert(nobjs[j].int() == len(gtmasks))
                    nos += nobjs[j].int()
                    for m in gtmasks :
                        r.append(int(m.sum()))
                #EKOX(typ)
                ne += len(images)
            EKOX(len(r))
            EKOX(len([e for e in r if e < 10000]))
            EKOX(len([e for e in r if e > 350000]))
            #n, bins, patches = plt.hist(r, 1000); plt.show()
            EKOX(ne)
            EKOX(nos)
            
        scores, losses, losses_bce = [], [], []
        
        for era in range(self.ERAS) :
            EKOX((era, self.ERAS))
            train_set, val_set = torch.utils.data.random_split(ds_train, [TS, train_len - TS])

            dl_train = DataLoader(dataset=subset(train_set, self.SUBSET), 
                                  batch_size=self.BATCH_SIZE, 
                                  drop_last=True, 
                                  num_workers=self.WORKERS, 
                                  shuffle=False) # pin_memory=True, 
            dl_val = DataLoader(dataset=subset(val_set, self.SUBSET), 
                                batch_size=self.BATCH_SIZE, 
                                num_workers=self.WORKERS, 
                                shuffle=False) # pin_memory=True, 
            #EKOX(len(dl_train) * self.BATCH_SIZE)
            #EKOX(len(dl_val) * self.BATCH_SIZE)
            n_batches = len(dl_train)
            # get a batch from the dataloader
            batch = next(iter(dl_train))
            images, masks, nobjs, typ, maskC = batch
            #EKOX(TYPE(maskC))

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

            def iouf(a, b) :
                inter = a * b
                union = a + b - inter
                iou = inter.sum() / (union.sum() + 1)
                return iou

            def fff(probability_mask, pnob, nob, dsz, threshold) :
                #EKOX(TYPE(probability_mask))
                def fff1(probability_mask, pnob, nob, dsz, threshold) :
                    probability_mask = cv2.resize(probability_mask, dsize=dsz, interpolation=cv2.INTER_LINEAR)
                    predictions, nobjs_ = post_process(probability_mask, dsize=dsz, threshold=threshold)
                    return predictions
                fff1p = partial(fff1, pnob=pnob, nob=nob, dsz=dsz, threshold=threshold) 

                ll = [ fff1p(pm) for pm in probability_mask]
                pms = [ x for pred in ll for x in pred]
                d1 = np.abs(len(pms) - nob.detach().cpu().numpy())
                d2 = np.abs(nob.cpu().numpy() - pnob.detach().cpu().numpy())
                return d1, d2, pms

            criterion = MixedLoss(10.0, 2.0)                    
            #criterion = torch.nn.CrossEntropyLoss()

            if self.args.model == "deeplab" :
                criterion = torch.nn.BCELoss()
            bce = torch.nn.BCEWithLogitsLoss()
            model.train()

            for epoch in range(1, self.EPOCHS + 1):
                EKOT(f"Starting epoch:  {epoch} / {self.EPOCHS}")
                #EKOX((era, self.ERAS))
                running_loss = 0.0
                running_loss2 = 0.0
                running_loss_bce = 0.0
                optimizer.zero_grad()

                for batch_idx, batch in self.tqdmf(enumerate(dl_train), total=len(dl_train)):
                    # Predict
                    images, masks, nobjs, typ, masksC = batch
                    images, masks, nobjs, typ, masksC = images.cuda(),  masks.cuda(), nobjs.cuda().float(), typ.cuda(), masksC.cuda()

                    #nobjs = nobjs.log() / 6 # pour rammener ce nombre entre 0 et 1
                    nobjs = (nobjs - meanNobjs) / stdNobjs
                    masks = masks # .round().long()

                    B, C, H, W = masks.shape

                    try :
                        outputs, pnobjs, ptyp = model(images)
                        #pnob = torch.clamp(pnob, 0, 10).T
                        #outputs = outputs.view(self.BATCH_SIZE, -1)
                        #masks = masks.view(self.BATCH_SIZE, -1)
                        pnobjs = (pnobjs * stdNobjs) + meanNobjs
                        """
                        loss1 = criterion(outputs, masks)
                        loss2 = nn.functional.l1_loss(pnob, nobjs)
                        loss_bce = bce(outputs, masks)
                        #EKOX((ptyp, typ))
                        loss3 = 0 #nn.functional.cross_entropy(ptyp, typ)

                        #EKOX(loss1.item())
                        #EKOX(loss2.item())
                        #EKOX(loss3.item())
                        
                        loss = loss1 + loss2 + loss3
                        """
                        #n, bins, patches = plt.hist(cannyi.cpu().numpy().flatten()); plt.show()
                        #pcanny = canny(outputs)[0]
                        #EKOX(outputs)
                        #EKOX(pcanny)
                        #EKOX(TYPE(outputs))

                        # fuse the channels
                        outputs1 = outputs.sum(dim=1, keepdim=True) #
                        #EKOX(TYPE(outputs1))
                        #EKOX(TYPE(masks))


                        if era >= 9999 :
                            preds = torch.sigmoid(outputs)
                            EKOX(TYPE(preds))
                            pnobjs = pnobjs[:,0]
                            #EKOX(pnobjs.T.detach().cpu().numpy())                
                            #pnobjs = (pnobjs * 6).exp() # inversion de la normalization
                            pnobjs = (pnobjs * stdNobjs) + meanNobjs
                            for jnoimage in range(B) :
                                masksCnp = masksC.cpu().numpy()
                                gtmasks,  _ = extractMasks(masksCnp[jnoimage, 0], dsize=DSIZE, min_size=1)
                                EKOX(len(gtmasks))
                                EKOX(TYPE(gtmasks))
                                B, C, _, _ = preds.shape
                                preds1 = preds.detach().cpu().numpy()
                                ll = []
                                for nc in range(C):
                                    preds2 = preds1[jnoimage, nc]
                                    _, _, pmasks = fff(preds2, pnobjs[jnoimage], nobjs[jnoimage], dsz=DSIZE, threshold=0.5)
                                    EKOX(TYPE(pmasks))
                                    ll += [ (nc, ip, ig, iouf(p,g)) for ip, p in enumerate(pmasks) for ig, g in enumerated(gtmasks) ]
                                EKO()
                                #mat = np.zeros((len(gtmasks), C * len(


                        # fused mask == gt
                        loss1 =  bce(outputs1, masks)

                        # superposition minimal
                        loss2 =  outputs.prod(dim=1).abs().mean()
                        
                        # maximize spatial variance
                        ps = outputs.sum(dim=(2,3))
                        vrnc = torch.var(outputs, dim=(2,3)).sqrt().mean() / 10 # H / W
                        loss3 = - vrnc
                        
                        # minimize mean over the classes bitplanes : cell equi distributed
                        mn = torch.mean(outputs, dim=(2,3))
                        vmn = torch.var(mn).sqrt().mean()
                        loss4 = vmn




                        #loss =  bce(pcanny, cannyi)
                        loss = loss1 + loss2 + loss3 + loss4
                        #EKOX((loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss.item()))
                        # Back prop
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        running_loss += loss.item()
                        #running_loss2 += loss2.item()
                        #running_loss_bce += loss_bce.item()
                    except Exception as e  :
                        EKOX(e)
                        if self.debug : raise(e)
                        pass
                epoch_loss = running_loss / n_batches
                EKOT(f"Epoch: {epoch} - Train Loss {epoch_loss:.4f}")
                EKOX(running_loss2 / n_batches)
                EKOX(running_loss_bce / n_batches)
                losses.append( epoch_loss)
                losses_bce.append( running_loss_bce / n_batches)
            EKOT("saving")
            torch.save(model.state_dict(), self.args.pts)
            #EKO()
            
            model.eval()

            def getimage(x, nb=0) :
                if isinstance(x, torch.Tensor) : x = x.cpu().detach().numpy()
                x = x[nb,0]
                x = np.dstack((x,x,x)) * 255
                x = x.astype(int)
                return x

            # ## Postprocessing: separate different components of the prediction mask


            def fbatch(inobatch, batch, th=0.5, threshold=0.5) :
                images, masks, nobjs, typ, masksC = batch

                b_size, _, _, _ =  images.shape

                preds, pnobjs, typ = model(images.cuda())
                pnobjs = pnobjs[:,0]
                preds = torch.sigmoid(preds)
                #EKOX(pnobjs.T.detach().cpu().numpy())                
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
                masksC = masksC.cpu().numpy()

                preds1 = preds #torch.sigmoid(preds)
                #EKOX(TYPE(preds1))
                preds1 = preds1.detach().cpu().numpy()[:, :, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
                
                _dsize = (224, 224)

                _dsize = DSIZE


                fffp = partial(fff, dsz = _dsize, threshold=threshold) 

                ppp1 = [ fffp(*e) for e in zip(preds1, pnobjs, nobjs)]
                ppp = [ (e[0], e[1]) for e in ppp1]
                ppp = np.asarray(ppp)
                #EKOX(TYPE(ppp))
               
                preds = (predsP > th).astype(int)
                preds6 = (predsP > 0.6).astype(int)
                preds4 = (predsP > 0.4).astype(int)
 
                inter = preds * masks
                union = preds + masks - inter
                iou = inter.sum(axis=(1,2,3)) / (union.sum(axis=(1,2,3)) + 1)

                nono = (pnobjs.cpu() - nobjs.cpu()).abs().detach().numpy()
                #raise Exception("a")
                
                predsQ = predsP[:, 0, :, :]
                def cnob(probability_mask) :
                    try:
                        #if probability_mask.shape != IMAGE_RESIZE:
                        #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                        probability_mask = cv2.resize(probability_mask, dsize=DSIZE, interpolation=cv2.INTER_LINEAR)
                        predictions, nobjs = post_process(probability_mask)
                        #EKOX(nobjs)
                    except Exception as e:
                        EKOX(e)
                        pass
                    return nobjs

                def getColored(m) :
                    height, width = m[0].shape
                    maskColors = np.zeros((height, width, 3))
                    for i, em in enumerate(m):
                        maskColors += np.dstack((em, em, em)) * colors[i % max_objs]
                    res = np.transpose(maskColors, axes=(1, 0, 2))
                    return res

                def batchprecf(jnoimage) :
                    """
                    la j ieme image du bactch
                    """
                    gtmasks,  _ = extractMasks(masksC[jnoimage, 0], dsize=_dsize, min_size=1)
                    #EKOX(len(gtmasks)) 
                    #EKOX(nobjs[j])
                    #EKOX(TYPE(preds1))
                    _, _, pmasks = fffp(preds1[jnoimage], pnobjs[jnoimage], nobjs[jnoimage])

                    #EKO()
                    pmasks = torch.tensor(pmasks).cuda()
                    gtmasks = torch.tensor(gtmasks).cuda()
                    
                    #EKOX(TYPE(gtmasks))
                    #EKOX(TYPE(pmasks))

                    dico = {}
                    #EKO()
                    for i, pm in enumerate(pmasks) :
                        for j, gt in enumerate(gtmasks) :
                            dico[(i,j)] = iouf(pm, gt)
                    #EKO()


                    def precf(th) :
                        #EKO()
                        def count(th, a, b, sw, cond) :
                            def iouf2(ix, iy) : return dico[ (iy, ix) ] if sw else dico[ (ix, iy) ] 
                            n = 0
                            #EKOX(len(a))
                            for ix, x in enumerate(a) :
                                #EKOX([ iouf(x,y) for y in b ])
                                lll = [ y for iy, y in enumerate(b) if iouf2(ix, iy) > th]
                                if (len(lll) > 0) == cond : n += 1
                            return n
                        tp = count(th, pmasks, gtmasks, False, True)
                        fp = count(th, pmasks, gtmasks, False, False)
                        fn = count(th, gtmasks, pmasks, True, False)

                        if inobatch == 0 and jnoimage == 0 :
                            EKOX((th, tp, fp, fn, len(gtmasks), len(pmasks)))

                            #EKOX(TYPE(gtmasks))
                            #EKOX(TYPE(pmasks))
                            
                            EKOI(getColored(gtmasks.cpu().numpy()))

                            if len(pmasks) > 0 :
                                EKOI(getColored(pmasks.cpu().numpy()))

                        p = 1 if len(gtmasks) == 0 else tp / (tp + fp + fn)
                        return p
                    #EKO()
                    precfs = [ precf(th) for th in SWEEP]
                    avprec = np.mean(precfs)
                    #EKO()
                    return avprec, precfs

                batchprecfs = [batchprecf(j) for j in range(b_size)]
                avprec = np.asarray([e[0] for e in batchprecfs])

                #EKOX([ cnob(probability_mask) for probability_mask in predsQ])
                #EKOX(pnobjs.T.detach().cpu().numpy())
                #EKOX([ cnob(probability_mask) for probability_mask in masks[:,0,:,:]])
                #EKOX(nobjs.T)
                if inobatch == 0 : 
                    EKOX(TYPE(ppp1[0][2]))
                    EKOX(TYPE(getimage(masks)))
                    ii = np.hstack((
                        (images[0].permute((1,2,0)).detach().cpu().numpy() * 255).astype(int),
                        getimage(masks),
                        getimage(preds),
                        getimage(preds6),
                        getimage(preds4),
                        getimage(inter),
                        getimage(union)))
                    EKOI(ii)
                    EKOX(batchprecfs[0][1])
                    EKOX(avprec[0])
                    EKOX((iou[0], pnobjs[0].item(), nobjs[0], len(ppp1[0][2])))
                    pass

                rr = np.asarray((iou, nono, ppp[:,0], avprec))
                return rr

            for thld in [ 0.4, 0.5, 0.6 ] :
                EKOX(thld)
                lflf = [ fbatch(i, batch, threshold=thld) for i, batch in enumerate(self.tqdmf(dl_val))]
                lflf = np.hstack(lflf)
                EKOT("iou, #objs pred error, #nmb actual seg error")
                res = np.mean(lflf, axis=1) 
                EKOX((thld, res))
            scores.append(res[3])

            with open("losses.txt", "w") as fd :  fd.write('\n'.join([str(e) for e in losses]))
            with open("losses_bce.txt", "w") as fd :  fd.write('\n'.join([ str(e) for e in losses_bce]))
            with open("scores.txt", "w") as fd :  fd.write('\n'.join([str(e) for e in scores]))
            

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

        def remove_overlapping_pixels(mask, other_masks):
            for other_mask in other_masks:
                if np.sum(np.logical_and(mask, other_mask)) > 0:
                    mask[np.logical_and(mask, other_mask)] = 0
            return mask

        # I am not sure if they adapt the sample submission csv or only the test folder
        # I am using the test folders as the ground truth for the images to predict, which should be always right
        # The sample csv is ignored
        pd.read_csv(SAMPLE_SUBMISSION)

        # ### Predict & submit



        submission = []
        for i, batch in enumerate(self.tqdmf(dl_test)):
            preds, pnobjs, typ = model(batch['image'].cuda())

            #pnobjs = (pnobjs * 6).exp() # inversion de la normalization
            pnobjs = (pnobjs * stdNobjs) + meanNobjs

            preds = torch.sigmoid(preds)
            preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
            for image_id, probability_mask, nob in zip(batch['id'], preds, pnobjs):
                try:
                    #if probability_mask.shape != IMAGE_RESIZE:
                    #    probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
                    probability_mask = cv2.resize(probability_mask, dsize=DSIZE, interpolation=cv2.INTER_LINEAR)
                    predictions, nobjs = post_process(probability_mask)                        
                    previous_masks = []
                    #EKOX(nob)
                    #EKOX(len(predictions))
                    for prediction in predictions:
                        #plt.imshow(prediction)
                        #plt.show()
                        try:
                            binary_mask = prediction
                            binary_mask = remove_overlapping_pixels(binary_mask, previous_masks)
                            previous_masks.append(binary_mask)
                            submission.append((image_id, rle_encoding(binary_mask)))
                        except Exception as e1:
                            EKOX(e)
                            print("Error in RL encoding")
                except Exception as e:
                    print(f"Exception for img: {image_id}: {e}")
                    submission.append((image_id, ""))
                    if self.debug : raise(e)

        df_submission = pd.DataFrame(submission, columns=['id', 'predicted'])
        df_submission.to_csv('submission_%s_%04d.csv' % (prefix, era), columns=['id', 'predicted'], index=False)
        df_submission.to_csv('submission.csv', columns=['id', 'predicted'], index=False)



            #EKOX(df_submission.head())

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:21:23.106890Z","iopub.execute_input":"2021-11-07T00:21:23.107637Z","iopub.status.idle":"2021-11-07T00:22:11.947180Z","shell.execute_reply.started":"2021-11-07T00:21:23.107595Z","shell.execute_reply":"2021-11-07T00:22:11.946462Z"}}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, default="")
    parser.add_argument('--debug', type=Bool, default=False)
    parser.add_argument('--model', default="unet")
    parser.add_argument('--pts', default="model.pts")
    parser.add_argument('-f')
    args = parser.parse_known_args()[0]
    EKOX(args)
    fs = Cell(args)
    if True : #with autograd.detect_anomaly():
        fs.run()
    EKOT("done")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:12:56.522788Z","iopub.execute_input":"2021-11-07T00:12:56.523044Z","iopub.status.idle":"2021-11-07T00:12:57.233655Z","shell.execute_reply.started":"2021-11-07T00:12:56.523015Z","shell.execute_reply":"2021-11-07T00:12:57.232772Z"}}
#!ls

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-11-07T00:12:57.236243Z","iopub.execute_input":"2021-11-07T00:12:57.236805Z","iopub.status.idle":"2021-11-07T00:12:59.386024Z","shell.execute_reply.started":"2021-11-07T00:12:57.236763Z","shell.execute_reply":"2021-11-07T00:12:59.384965Z"}}
if False :
    EKOX("coucou")
    EKOX(__name__)

    with open("text.txt", "w") as fd :
        fd.write("text")
"""
!pwd
!ls
!ls *.cpt
"""
# %% [code] {"jupyter":{"outputs_hidden":false}}

