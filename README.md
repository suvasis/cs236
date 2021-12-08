cs236 project code and steps

1. Data peparation

    //From the report data[5] download the following 12 million image caption pairs
    //Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset
      For Automatic Image Captioning

     create a folder cs236/cc12m

   a) cd cs236/data
   
      python CC12M_downloads.py
      
   b) python download_images.py
   
   c) preprocess_map_image_to_text.py
   
2. Encoding the image using pretrained VQGAN

   cd ../code
   
   python vegan_encoding.py
   
2. Running Trainging and evaluation

 
   change the training and evaluation folder directory in do_run.sh
   
   ./do_run.sh
   
 4. clone the git https://github.com/lzhbrian/metrics to calculate 
 
IS and FID score


# calc IS score on custom images in a folder/

python is_fid_pytorch.py --path foldername/


# calc IS, FID score on custom images in a folder/, compared to CIFAR10 (given precalculated stats)

python is_fid_pytorch.py --path foldername/ --fid res/stats_pytorch/fid_stats_cifar10_train.npz



