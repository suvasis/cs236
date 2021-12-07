import sys
import os
from datetime import datetime
import pandas as pd
import contexttimer
from urllib.request import urlopen
import requests
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from multiprocessing import Pool
from tqdm import tqdm
import logging


# Setup
#run1
logging.basicConfig(filename='download1.log', filemode='w', level=logging.INFO)
#run2
#logging.basicConfig(filename='download2.log', filemode='w', level=logging.INFO)
#run3
#logging.basicConfig(filename='download3.log', filemode='w', level=logging.INFO)
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


# # For downloading SVG images (I can't get this to work)
# from io import BytesIO
# import cairosvg

#%% 
# Load data
print(f'Starting to load at {datetime.now().isoformat(timespec="minutes")}')
with contexttimer.Timer(prefix="Loading from tsv"):
    df = pd.read_csv('./cc12m/cc12m.tsv', delimiter='\t', header=None)

url_to_idx_map = {url: index for index, url, caption in df.itertuples()}
#12420354
print(f'Loaded {len(url_to_idx_map)} urls')

#%% 
df.head()
print(df.head())
#%% 

# Note: it seems that there are no SVG images
df.sample(10000)[1].str.contains('.svg').sum()


#%% 
# Resize function
def resize(img):
    max_size_of_short_side = 512
    if min(img.size) > max_size_of_short_side:
        img = TF.resize(img, size=max_size_of_short_side, interpolation=Image.LANCZOS)
    return img

#run1 images1
base_dir = os.path.join(os.getcwd(), 'images1')
#run2 images2
#base_dir = os.path.join(os.getcwd(), 'images3')
#run3 images3
#base_dir = os.path.join(os.getcwd(), 'images3')
print(base_dir)

def process(item):
    url, image_id = item
    try:
        base_url = os.path.basename(url)  # extract base url
                stem, ext = os.path.splitext(base_url)  # split into stem and extension
        filename = f'{image_id:08d}---{stem}.jpg'  # create filename
        filepath = os.path.join(base_dir, filename)  # concat to get filepath
        if not os.path.isfile(filepath):
            # if filepath.endswith('.svg'):
            #     raise NotImplementedError()
            #     image_bytes = BytesIO()  # create a bytestream
            #     cairosvg.svg2png(url=url, write_to=image_bytes)  # convert svg into image
            # else:
            req = requests.get(url, stream=True, timeout=1, verify=False).raw
            image = Image.open(req).convert('RGB')
            if min(image.size) > 512:
                image = TF.resize(image, size=512, interpolation=Image.LANCZOS)
            # image = resize(image)  # resize PIL image
            image.save(filepath)  # save PIL image
            #download.log should have the filenames
            #logging.info("0 >>> ")
            tolog = filename + " " + url
            logging.info(tolog)
            #logging.info(" 00 >>> ")
            #logging.info(url)
    except Exception as e:
        logging.info(" ".join(repr(e).splitlines()))
        logging.error(url)

#%% 
#for i, item in enumerate(tqdm(url_to_idx_map.items(), total=len(url_to_idx_map))):
#    process(item)
#    if i > 100:
#        break

# Use multiprocessing for speed
list_of_items = list(url_to_idx_map.items())
print(">>> 1: ",len(list_of_items))
#list_of_items = list_of_items[10_000_000:]
#>>>>>>>>>  2420354

#>>> 1:  12420354
# >>>>>>> 2:  10000000
#list_of_items = list_of_items[:10_000_000]
#1st 5 mil download1.log run1 
list_of_items = list_of_items[500_000:5_000_000]

# 5mil to 10 mil download2.log run2
#list_of_items = list_of_items[5_000_001:10_000_000]
#rest  2420354 download3.log run3
#list_of_items = list_of_items[10_000_001:]

print(" >>>>>>> 2: ",len(list_of_items))
#logging.info("1 >>>  ")
#logging.info(list_of_items[0])
#logging.info("2 >>>  ")
#logging.info(list_of_items[1])
with Pool(128) as p:
    r = list(tqdm(p.imap(process, list_of_items), total=len(list_of_items)))
    print('DONE')


