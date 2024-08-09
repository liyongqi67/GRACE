from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
import json
import os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--v_dim', type=int, default=768)
parser.add_argument('--dataset', type=str, default='NQ')
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--k', type=int, default= 30)
parser.add_argument('--c', type=int, default= 30)

parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
args = parser.parse_args()




from sentence_transformers import SentenceTransformer, util
from PIL import Image


#Load CLIP model
model = SentenceTransformer('clip-ViT-L-14')



#read image names
image_name_list = []
array = np.zeros((300000, 768), dtype=np.float64)
with open(args.dataset, 'r') as f:
    data = json.load(f)['images']
    for original_sample_data in tqdm(data):
        if original_sample_data['split'] !='restval':
            image_name_list.append(original_sample_data['filename'])
            # img_emb = model.encode(Image.open(os.path.join(args.image_dir, original_sample_data['filename'])))
            text_emb = model.encode(original_sample_data['sentences'][0]['raw'])
            text_emb = text_emb.astype(np.float64)
            #Compute cosine similarities 
            cos_scores = util.cos_sim(array, text_emb)
            print(cos_scores)

# # #read image names
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained('anas-awadalla/mpt-1b-redpajama-200b')
# text_tokenizer = AutoTokenizer.from_pretrained(
#     'anas-awadalla/mpt-1b-redpajama-200b',
#     local_files_only=True,
# )
# inputs = text_tokenizer(["Today is"], return_tensors="pt")
# with open(args.dataset, 'r') as f:
#     data = json.load(f)['images']
#     for original_sample_data in tqdm(data):
#         if original_sample_data['split'] !='restval':
#             model.generate(
#     **inputs,
#     max_new_tokens=2,
#     num_beams=1,
#     num_return_sequences=1,
# )
            