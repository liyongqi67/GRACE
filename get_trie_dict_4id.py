from tqdm import tqdm
import re
import json
from utils import Trie
import pickle
from open_flamingo import create_model_and_transforms
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_dir",
    type=str,
    help="Pass in the directory where the output shards (as tar files) will be written to.",
)
arg_parser.add_argument(
    "--json_file",
    type=str,
    help="image-caption pairs json_file",
)
arg_parser.add_argument(
    "--image_dir",
    type=str,
    help="Pass in the directory where the images have been downloaded to.",
)
arg_parser.add_argument(
    "--image_name2id_dict",
    type=str,
)
args = arg_parser.parse_args()
with open(args.image_name2id_dict, 'rb') as f:
    id2image_name_dict = pickle.load(f)
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)
id_dict_test_set = {}
with open(args.json_file, 'r') as f:
    data = json.load(f)['images']
    for original_sample_data in data:
        if original_sample_data['split'] =='test':
            id_dict_test_set[str(id2image_name_dict[original_sample_data['filename']])] = 1




end_of_chunk_id = tokenizer.encode("<|endofchunk|>")[0]
print('end_of_chunk_id',end_of_chunk_id)
caption_sequence = []
for caption in tqdm(id_dict_test_set):
    input_ids = tokenizer.encode(
    "<image>id "+ caption,
    add_special_tokens=True,
    max_length=10,
    truncation=True)
    caption_sequence.append(input_ids+[end_of_chunk_id])

decoder_trie = Trie(caption_sequence)
print("decoder_trie len %s", decoder_trie.len)
with open(args.output_dir, 'wb') as f:
    pickle.dump(decoder_trie.trie_dict, f)