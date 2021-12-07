#We process a tsv file with image_file and caption fields, and add a vqgan_indices column with indices extracted from a VQGAN-JAX model.

import io

import requests
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
#I found an easy fix which will make the package compliant with torchvision=0.8.1. Just remove InterpolationMode import and code related to it. torchvision.transforms.Resize uses bilinear interpolation by default, so there's no need to specify it directly.
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader

import jax
from jax import pmap

#print("text")

#########################MODEL#################
print('VQGAN-JAX model')
#('VQGAN-JAX model')

from vqgan_jax.modeling_flax_vqgan import VQModel
model = VQModel.from_pretrained("flax-community/vqgan_f16_16384")

#########################DATASET#################
print('Dataset')
#('Dataset')

from project.dataset import *

#(squad) minasm@lambda-quad:~/suvasis/tools/cs236_project/code/encoding$ ls ../../data/cc12m/
#cc12m_clean_list.tsv  cc12m.tsv 

#####################################################
#cc12m_images = '../../data/images'
#cc12m_list = '../../data/cc12m/cc12m_clean_list_test.tsv' #images-list-clean.tsv'
## cc12m_list = '/data/CC12M/images-10000.tsv'
#cc12m_output = '../../data/cc12m/images-encoded.tsv'
######################################################
#(base) minasm@lambda-quad:~/suvasis/tools/cs236_project/data/cc12m/allimages$ ls
#cc12m_clean_list_10m.tsv  images  images1  images10m  images2  images3  images5m  images5m_10m  syncfiles.py
#####################################################

#10 mil to 12.mil
#cc12m_images = '../../data/cc12m/allimages/images3'
#cc12m_list = '../../data/cc12m/allimages/cc12m_clean_list_10m.tsv' #images-list-clean.tsv'
## cc12m_list = '/data/CC12M/images-10000.tsv'
#cc12m_output = '../../data/cc12m/images-encoded_10m.tsv'

#5 mil to 10.mil
cc12m_images = '../../data/cc12m/allimages/images2'
cc12m_list = '../../data/cc12m/allimages/cc12m_clean_list_5m10m.tsv' #images-list-clean.tsv'
# cc12m_list = '/data/CC12M/images-10000.tsv'
cc12m_output = '../../data/cc12m/images-encoded_5m10m.tsv'

#print(" 1. ",cc12m_images)
#print(" 2. ",cc12m_list)
#print(" 3. ",cc12m_output)

image_size = 256
def image_transform(image):
    s = min(image.size)
    r = image_size / s
    s = (round(r * image.size[1]), round(r * image.size[0]))
    image = TF.resize(image, s, interpolation=InterpolationMode.LANCZOS)
    #use Resize instead
    #Resize uses bilinear interpolation by default
    #print(" image size ", s)
    #image = T.Resize(s)
    #this is throwing error dont have outputsize
    image = TF.center_crop(image, output_size = 2 * [image_size])
    #image = T.CenterCrop( 2 * [image_size])
    image = torch.unsqueeze(T.ToTensor()(image), 0)
    image = image.permute(0, 2, 3, 1).numpy()
    return image

dataset = CaptionDataset(
    images_root=cc12m_images,
    captions_path=cc12m_list,
    image_transform=image_transform,
    image_transform_type='torchvision',
    include_captions=False
)

#print(len(dataset))

#########################ENCODING#################
print('Encoding')
#('Encoding')

def encode(model, batch):
#     print("jitting encode function")
    _, indices = model.encode(batch)
    return indices

def superbatch_generator(dataloader, num_tpus):
    iter_loader = iter(dataloader)
    for batch in iter_loader:
        #print(">>>>> ", batch)
        superbatch = [batch.squeeze(1)]
        try:
            for b in range(num_tpus-1):
                batch = next(iter_loader)
                if batch is None:
                    break
                # Skip incomplete last batch
                if batch.shape[0] == dataloader.batch_size:
                    superbatch.append(batch.squeeze(1))
        except StopIteration:
            pass
        superbatch = torch.stack(superbatch, axis=0)
        yield superbatch


import os

def encode_captioned_dataset(dataset, output_tsv, batch_size=32, num_workers=16):
    if os.path.isfile(output_tsv):
        print(f"Destination file {output_tsv} already exists, please move away.")
        return

    #num_tpus = 8
    #there is no tpus set it to 1
    num_tpus = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    superbatches = superbatch_generator(dataloader, num_tpus=num_tpus)

    p_encoder = pmap(lambda batch: encode(model, batch))

    # We save each superbatch to avoid reallocation of buffers as we process them.
    # We keep the file open to prevent excessive file seeks.
    with open(output_tsv, "w") as file:
        iterations = len(dataset) // (batch_size * num_tpus)
        #print(">>>> iterations ", iterations, " batch size ", batch_size, " num_tpus ", num_tpus)
        for n in tqdm(range(iterations)):
            superbatch = next(superbatches)
            encoded = p_encoder(superbatch.numpy())
            encoded = encoded.reshape(-1, encoded.shape[-1])

            # Extract fields from the dataset internal `captions` property, and save to disk
            start_index = n * batch_size * num_tpus
            end_index = (n+1) * batch_size * num_tpus
            paths = dataset.captions["image_file"][start_index:end_index].values
            captions = dataset.captions["caption"][start_index:end_index].values
            #print(">>>>>>>> ",paths, " >>>1 >>",captions)
            #base_url = os.path.basename(url)  # extract base url
            #stem, ext = os.path.splitext(base_url)  # split into stem and extension
            #filename = f'{image_id:08d}---{stem}.jpg'
            encoded_as_string = list(map(lambda item: np.array2string(item, separator=',', max_line_width=50000, formatter={'int':lambda x: str(x)}), encoded))
            batch_df = pd.DataFrame.from_dict({"image_file": paths, "caption": captions, "encoding": encoded_as_string})
            batch_df.to_csv(file, sep='\t', header=(n==0), index=None)


###################################ENCODING############################
#encode_captioned_dataset(dataset, cc12m_output, batch_size=64, num_workers=16)
encode_captioned_dataset(dataset, cc12m_output, batch_size=2, num_workers=1)

exit(0)
