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
    "--num_files_per_shard",
    type=int,
    default=5000,
)
args = arg_parser.parse_args()
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1
)
caption_dict_test_set = {}
with open(args.json_file, 'r') as f:
    data = json.load(f)['images']
    for original_sample_data in data:
        if original_sample_data['split'] =='test':
            for sentence in original_sample_data['sentences']:
                if sentence["raw"].strip() not in caption_dict_test_set:
                    caption_dict_test_set[sentence["raw"].strip()] = 1



image_token_id = tokenizer.encode("<image>")[0]
print('image_token_id',image_token_id)
end_of_chunk_id = tokenizer.encode("<|endofchunk|>")[0]
print('end_of_chunk_id',end_of_chunk_id)
caption_sequence = []
for caption in tqdm(caption_dict_test_set):
    input_ids = tokenizer.encode(
    caption,
    add_special_tokens=True,
    max_length=64,
    truncation=True)
    caption_sequence.append([image_token_id]+input_ids+[end_of_chunk_id])

decoder_trie = Trie(caption_sequence)
print("decoder_trie len %s", decoder_trie.len)
with open(args.output_dir, 'wb') as f:
    pickle.dump(decoder_trie.trie_dict, f)