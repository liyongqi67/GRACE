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
X = [] #visual emb list
with open(args.dataset, 'r') as f:
    data = json.load(f)['images']
    for original_sample_data in tqdm(data):
        if original_sample_data['split'] !='restval':
            image_name_list.append(original_sample_data['filename'])
            img_emb = model.encode(Image.open(os.path.join(args.image_dir, original_sample_data['filename'])))
            X.append(img_emb)
        

X = np.array(X)
print(X.shape)
np.save('image_emb.npy', X)

new_id_list = []

kmeans = KMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=args.seed, tol=1e-7)

mini_kmeans = MiniBatchKMeans(n_clusters=args.k, max_iter=300, n_init=100, init='k-means++', random_state=3,
                              batch_size=1000, reassignment_ratio=0.01, max_no_improvement=20, tol=1e-7)


def classify_recursion(x_data_pos):
    if x_data_pos.shape[0] <= args.c:
        if x_data_pos.shape[0] == 1:
            return
        for idx, pos in enumerate(x_data_pos):
            new_id_list[pos].append(idx)
        return

    temp_data = np.zeros((x_data_pos.shape[0], args.v_dim))
    for idx, pos in enumerate(x_data_pos):
        temp_data[idx, :] = X[pos]

    if x_data_pos.shape[0] >= 1e3:
        pred = mini_kmeans.fit_predict(temp_data)
    else:
        pred = kmeans.fit_predict(temp_data)

    for i in range(args.k):
        pos_lists = []
        for id_, class_ in enumerate(pred):
            if class_ == i:
                pos_lists.append(x_data_pos[id_])
                new_id_list[x_data_pos[id_]].append(i)
        classify_recursion(np.array(pos_lists))

    return

print('Start Clustering')
pred = mini_kmeans.fit_predict(X)
print(pred.shape)   #int 0-9 for each vector
print(mini_kmeans.n_iter_)

for class_ in pred:
    new_id_list.append([class_])

print('Start Recursively Clustering...')
for i in range(args.k):
    print(i, "th cluster")
    pos_lists = [];
    for id_, class_ in enumerate(pred):
        if class_ == i:
            pos_lists.append(id_)
    classify_recursion(np.array(pos_lists))


#print(new_id_list[100:200])
image_name2structured_id_dict = {}
structured_id2image_name_dict = {}
for i in range(len(image_name_list)):
    a=[]
    for j in range(len(new_id_list[i])):
        a.append(str(j)+"_"+str(new_id_list[i][j]))
    image_name2structured_id_dict[image_name_list[i]] = '-'.join(a)
    structured_id2image_name_dict['-'.join(a)] = image_name_list[i]

print('image_name2structured_id_dict len', len(image_name2structured_id_dict))
print('structured_id2image_name_dict len', len(structured_id2image_name_dict))
with open('image_name2structured_id_dict.pkl', 'wb') as f:
    pickle.dump(image_name2structured_id_dict, f)
with open('structured_id2image_name_dict.pkl', 'wb') as f:
    pickle.dump(structured_id2image_name_dict, f)


