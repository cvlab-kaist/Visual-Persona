#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2


from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .networks import init_model
from .utils.transforms import transform_logits
from .datasets.simple_extractor_dataset import SimpleFolderDataset


model_args = {
   'input_size': [512, 512],
    'num_classes': 18,
    'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
              'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
}

def load_parser_model(ckpt_path):
    model = init_model('resnet101', num_classes=model_args['num_classes'], pretrained=None)

    state_dict = torch.load_state_dict(ckpt_path)#['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    return model


def parse(model, image_path):

    input_size = model_args['input_size']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    dataset = SimpleFolderDataset(root=image_path, input_size=input_size, transform=transform)
    dataloader = DataLoader(dataset)

    body_part_idxs = [
        [1, 2, 3, 11],
        [4, 7, 14, 15, 17],
        [5, 6, 8, 12, 13],
        [9, 10]
    ]

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
          
            output = model(image.to(device))
            upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
            parsing_result = np.argmax(logits_result, axis=2)
            
            full_body_mask = Image.fromarray((1-(np.isin(parsing_result, [0, 16]).astype(np.uint8)).astype(np.uint8)) * 255)
            
            mask_list = [full_body_mask]
            for key, part_idx in enumerate(body_part_idxs):
                mask = Image.fromarray((np.isin(parsing_result, part_idx).astype(np.uint8)) * 255)
                mask_list.append(mask)
    
    return mask_list
