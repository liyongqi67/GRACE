import argparse
import json
import os
import uuid
import zipfile
from PIL import Image
import base64
from io import BytesIO

import braceexpand
import webdataset as wds
import pickle
import random
num=0

with open("/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/Openflamingo_format/coco/image_name2string_id_dict.pkl", 'rb') as f:
    image_name2id_dict = pickle.load(f)


with open("../data/dataset_coco.json", 'r') as f:
    data = json.load(f)['images']
    for original_sample_data in data:

        if original_sample_data['split'] =='test':
        	print(image_name2id_dict[original_sample_data['filename']])