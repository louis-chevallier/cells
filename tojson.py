import pandas as pd
import numpy as np
from tqdm import tqdm
import json,itertools

import utillc
from utillc import *

import cell13

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction

# From https://newbedev.com/encode-numpy-array-using-uncompressed-rle-for-coco-dataset
def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
def coco_structure(train_df):
    cat_ids = {name:id+1 for id, name in enumerate(train_df.cell_type.unique())}    
    cats =[{'name':name, 'id':id} for name,id in cat_ids.items()]
    images = [{'id':id, 'width':row.width, 'height':row.height, 'file_name':f'train/{id}.png'} for id,row in train_df.groupby('id').agg('first').iterrows()]
    annotations=[]
    EKOX(train_df.count())
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df.index)) :
        mk = rle_decode(row.annotation, (row.height, row.width))
        ys, xs = np.where(mk)
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        enc =binary_mask_to_rle(mk)
        seg = {
            'segmentation':enc, 
            'bbox': [int(x1), int(y1), int(x2-x1+1), int(y2-y1+1)],
            'area': int(np.sum(mk)),
            'image_id':row.id, 
            'category_id':cat_ids[row.cell_type], 
            'iscrowd':0, 
            'id':idx
        }
        annotations.append(seg)
    return {'categories':cats, 'images':images,'annotations':annotations}

input = cell13.input

EKO()
## run it on first three images for demonstration:
train_df = pd.read_csv(os.path.join(input, 'train.csv'))
all_ids = train_df.id.unique()

EKOX(TYPE(all_ids))
nnn = 9999999
val_sample = train_df[train_df.id.isin(all_ids[:100])]
train_sample = train_df[train_df.id.isin(all_ids[100:])]
root = coco_structure(val_sample)
with open(os.path.join(input, 'sartorius-cell-instance-segmentation-coco/annotations_train.json'), 'w', encoding='utf-8') as f:  
          json.dump(root, f, ensure_ascii=True, indent=4)
root = coco_structure(train_sample)
with open(os.path.join(input, 'sartorius-cell-instance-segmentation-coco/annotations_val.json'), 'w', encoding='utf-8') as f:  
          json.dump(root, f, ensure_ascii=True, indent=4)


