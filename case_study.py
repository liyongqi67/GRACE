from open_flamingo import create_model_and_transforms
import pickle
import torch


from PIL import Image
import requests

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

msd = model.state_dict()
old_state_dict = {}
for key in msd:
    old_state_dict[key] = msd[key].clone()


# grab model checkpoint from huggingface hub
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
model.load_state_dict(torch.load(checkpoint_path), strict=False)

msd = model.state_dict()
new_state_dict = {}
for key in msd:
    new_state_dict[key] = msd[key].clone()

for key in old_state_dict:
    if not (old_state_dict[key].cpu() == new_state_dict[key].cpu()).all():
        print('Diff in {}'.format(key))
    else:
        print('NO Diff in {}'.format(key))


# with open("../data/Openflamingo_format/structured_id2image_name_dict.pkl", 'rb') as f:
#     add_extra_id_tokens = pickle.load(f)
# id_token_dict = {}    
# for key in add_extra_id_tokens:
#     for s in key.split("-"):
#         id_token_dict[s]=1
# tokenizer.add_special_tokens({"additional_special_tokens": list(id_token_dict.keys())})
# model.lang_encoder.resize_token_embeddings(len(tokenizer))
# msd = model.state_dict()
# old_state_dict = {}
# for key in msd:
#     old_state_dict[key] = msd[key].clone()
# checkpoint = torch.load("../data/checkpoints/flicker30k_i2id27/epoch_1/mp_rank_00_model_states.pt",map_location="cpu")
# if "model_state_dict" in checkpoint:
#     checkpoint = checkpoint["model_state_dict"]
#     checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
# if "module" in checkpoint:
#     msd = checkpoint["module"]
# old_state_dict = {}
# for key in msd:
#     old_state_dict[key] = msd[key].clone()


# checkpoint = torch.load("../data/checkpoints/flicker30k_i2id27/epoch_4/mp_rank_00_model_states.pt",map_location="cpu")
# if "model_state_dict" in checkpoint:
#     checkpoint = checkpoint["model_state_dict"]
#     checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
# if "module" in checkpoint:
#     msd = checkpoint["module"]
# new_state_dict = {}
# for key in msd:
#     new_state_dict[key] = msd[key].clone()

# for key in old_state_dict:
#     if not (old_state_dict[key].cpu() == new_state_dict[key].cpu()).all():
#         print('Diff in {}'.format(key))
#     else:
#         print('NO Diff in {}'.format(key))
# model.load_state_dict(msd, strict=False)


# query_image = Image.open(
#         "../data/Flickr30K/flickr30k-images/1369162.jpg", 
# )
# """
# Step 2: Preprocessing images
# Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
#  batch_size x num_media x num_frames x channels x height x width. 
#  In this case batch_size = 1, num_media = 3, num_frames = 1,
#  channels = 3, height = 224, width = 224.
# """
# vision_x = [image_processor(query_image).unsqueeze(0)]
# vision_x = torch.cat(vision_x, dim=0)
# vision_x = vision_x.unsqueeze(1).unsqueeze(0)

# """
# Step 3: Preprocessing text
# Details: In the text we expect an <image> special token to indicate where an image is.
#  We also expect an <|endofchunk|> special token to indicate the end of the text 
#  portion associated with an image.
# """
# tokenizer.padding_side = "left" # For generation padding tokens should be on the left
# lang_x = tokenizer(
#     ["image numeric id<image>"],
#     return_tensors="pt",
# )

# """
# Step 4: Generate text 
# """
# generated_text = model.generate(
#     vision_x=vision_x,
#     lang_x=lang_x["input_ids"],
#     attention_mask=lang_x["attention_mask"],
#     max_new_tokens=25,
#     num_beams=5,
#     num_return_sequences=5
# )

# print("Generated text: ", tokenizer.batch_decode(generated_text))


# # checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
# # msd = torch.load(checkpoint_path)
# # msd = {k.replace("module.", ""): v for k, v in msd.items()}
# # old_state_dict = {}
# # for key in msd:
# #     old_state_dict[key] = msd[key].clone()




# # checkpoint_path = "/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2id9/checkpoint_0.pt"
# # msd = torch.load(checkpoint_path,map_location=torch.device('cpu'))['model_state_dict']
# # msd = {k.replace("module.", ""): v for k, v in msd.items()}
# # old_state_dict = {}
# # for key in msd:
# #     old_state_dict[key] = msd[key].clone()



# checkpoint_path = "/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2id10/checkpoint_0.pt"
# msd = torch.load(checkpoint_path,map_location=torch.device('cpu'))['model_state_dict']
# msd = {k.replace("module.", ""): v for k, v in msd.items()}
# new_state_dict = {}
# for key in msd:
#     new_state_dict[key] = msd[key].clone()

# for key in old_state_dict:
#     if not (old_state_dict[key].cpu() == new_state_dict[key].cpu()).all():
#         print('Diff in {}'.format(key))
#     else:
#         print('NO Diff in {}'.format(key))
# # model.load_state_dict(msd, strict=False)

# # from PIL import Image
# # import requests
# # import torch

# # """
# # Step 1: Load images
# # """

# # query_image = Image.open(
# #         "../data/Flickr30K/flickr30k-images/1369162.jpg", 
# # )
# # """
# # Step 2: Preprocessing images
# # Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
# #  batch_size x num_media x num_frames x channels x height x width. 
# #  In this case batch_size = 1, num_media = 3, num_frames = 1,
# #  channels = 3, height = 224, width = 224.
# # """
# # vision_x = [image_processor(query_image).unsqueeze(0)]
# # vision_x = torch.cat(vision_x, dim=0)
# # vision_x = vision_x.unsqueeze(1).unsqueeze(0)

# # """
# # Step 3: Preprocessing text
# # Details: In the text we expect an <image> special token to indicate where an image is.
# #  We also expect an <|endofchunk|> special token to indicate the end of the text 
# #  portion associated with an image.
# # """
# # tokenizer.padding_side = "left" # For generation padding tokens should be on the left
# # lang_x = tokenizer(
# #     ["image numeric id<image>"],
# #     return_tensors="pt",
# # )


# # """
# # Step 4: Generate text 
# # """
# # generated_text = model.generate(
# #     vision_x=vision_x,
# #     lang_x=lang_x["input_ids"],
# #     attention_mask=lang_x["attention_mask"],
# #     max_new_tokens=25,
# #     num_beams=5,
# #     num_return_sequences=5
# # )

# # print("Generated text: ", tokenizer.batch_decode(generated_text))

        
# # with open('/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id/image_name2id_dict.pkl', 'rb') as f:
# #     image_name2id_dict = pickle.load(f)
# # print(image_name2id_dict['1369162.jpg'])


# # def prefix_allowed_tokens_fn(batch_id, sent):
# #     return decoder_trie.get(sent.tolist())

# # """
# # Step 4: Generate text constrained
# # """
# # from utils import Trie
# # import pickle
# # with open("/home/share/yongqi/project/AutoregressiveImageRetrieval/code/caption_trie_test_set.pkl", 'rb') as f:
# #     decoder_trie = Trie.load_from_dict(pickle.load(f))
# # print("decoder_trie len %s", decoder_trie.len)

# # generated_text = model.generate(
# #     vision_x=vision_x,
# #     lang_x=lang_x["input_ids"],
# #     attention_mask=lang_x["attention_mask"],
# #     max_new_tokens=25,
# #     num_beams=5,
# #     num_return_sequences=5,
# #     prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
# # )

# # print("Generated text: ", tokenizer.batch_decode(generated_text))