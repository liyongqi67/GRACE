#!/bin/sh

## i2id scripts, structured_id
# CUDA_VISIBLE_DEVICES=2,3,4,5 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=1996 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_i2id 64 \
#     --train_num_samples_i2id 800000 \
#     --batch_size_t2id 64 \
#     --train_num_samples_t2id 800000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_i2id_sample=80w*3_1e-4" \
#     --precision bf16 \
#     --num_epochs 3 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#     --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id5/{000000000..00000006}.tar" \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --report_to_wandb


# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_i2id_sample=80w*3_1e-4/); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_sample=80w*3_1e-4/" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_i2id \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2structured_id_dict.pkl" \
#             --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl"
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed








##i2id and t2id, id is structured id, need to add_extra_id_tokens scripts
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_i2id 16 \
#     --train_num_samples_i2id 200000 \
#     --batch_size_t2id 48 \
#     --train_num_samples_t2id 600000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_dropout_wopseudo_query" \
#     --precision bf16 \
#     --num_epochs 5 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#     --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id5/{000000000..00000006}.tar" \
#     --t2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_t2id5_2/{000000000..000000029}.tar" \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --report_to_wandb


# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_dropout_wopseudo_query); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_dropout_wopseudo_query" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_t2id \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2structured_id_dict.pkl" \
#             --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#             --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#             --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id_trie_test_set.pkl"
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed


# i2id and id2caption, id is structured id, need to add_extra_id_tokens scripts,
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_i2id 16 \
#     --train_num_samples_i2id 200000 \
#     --batch_size_id2caption 48 \
#     --train_num_samples_id2caption 600000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_i2id_id2caption_sample=80w*5_1e-4" \
#     --precision bf16 \
#     --num_epochs 5 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#     --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id5/{000000000..00000006}.tar" \
#     --id2caption_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_id2caption/{000000000..000000029}.tar" \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --report_to_wandb


# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_i2id_id2caption_sample=80w*5_1e-4); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_id2caption_sample=80w*5_1e-4" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nnodes=1 --nproc_per_node=3 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_id2caption \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2structured_id_dict.pkl" \
#             --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#             --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" 
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed







# i2id and id2caption and caption2id, id is structured id, need to add_extra_id_tokens scripts,
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_i2id 8 \
#     --train_num_samples_i2id 200000 \
#     --batch_size_id2caption 24 \
#     --train_num_samples_id2caption 600000 \
#     --batch_size_t2id 24 \
#     --train_num_samples_t2id 600000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_i2id_id2caption_t2id_sample=140w*5_1e-4_dropout" \
#     --precision bf16 \
#     --num_epochs 5 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#     --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id5/{000000000..00000006}.tar" \
#     --id2caption_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_id2caption/{000000000..000000029}.tar" \
#     --t2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_t2id5/{000000000..000000030}.tar" \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --report_to_wandb


# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_i2id_id2caption_t2id_sample=140w*5_1e-4_dropout); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_id2caption_t2id_sample=140w*5_1e-4_dropout" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_t2id \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2structured_id_dict.pkl" \
#             --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#             --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl" \
#             --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id_trie_test_set.pkl"
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed



# i2id and t2id, id is semantic id, do not need to add_extra_id_tokens scripts
# CUDA_VISIBLE_DEVICES=2,3,4,5 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_i2id 16 \
#     --train_num_samples_i2id 200000 \
#     --batch_size_t2id 48 \
#     --train_num_samples_t2id 600000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_semanticID" \
#     --precision bf16 \
#     --num_epochs 5 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id7/{000000000..00000006}.tar" \
#     --t2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_t2id7/{000000000..000000029}.tar" \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --report_to_wandb
# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_semanticID); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_sample=80w*5_1e-4_semanticID" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_t2id \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2semantic_id_dict.pkl" \
#             --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/semantic_id2image_name_dict.pkl" \
#             --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/semantic_id_trie_test_set.pkl"
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed









#######automatic id with clip on flicker
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#     --model_family flamingo \
#     --cross_attn_every_n_layers 1 \
#     --dataset_resampled \
#     --batch_size_t2id 64 \
#     --train_num_samples_t2id 600000 \
#     --workers=4 \
#     --deepspeed \
#     --deepspeed_stage 3 \
#     --run_name "./checkpoints/deepspeed3_bf16_t2id_automaticID_CLIP_initial" \
#     --precision bf16 \
#     --num_epochs 5 \
#     --gradient_checkpointing \
#     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
#     --learning_rate 1e-4 \
#     --lr_scheduler linear \
#     --warmup_steps  500 \
#     --t2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_t2id8/{000000000..000000030}.tar" \
#     --new_class_embed \
#     --loss Classifier_loss \
#     --wandb_project Gen_Cross_Modal-Retrieval \
#     --delete_previous_checkpoint \
#     --report_to_wandb

# eval "$(conda shell.bash hook)"
# conda activate openflamingo
# for file in $(find ./checkpoints/deepspeed3_bf16_t2id_automaticID_CLIP_initial); do
#     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_t2id_automaticID_CLIP_initial" ]; then
#         echo $file
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
#             --vision_encoder_path ViT-L-14 \
#             --vision_encoder_pretrained openai\
#             --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
#             --cross_attn_every_n_layers 1 \
#             --checkpoint_path $file \
#             --results_file results.json \
#             --precision fp32 \
#             --batch_size 8 \
#             --eval_flickr_t2id_classifier \
#             --new_class_embed \
#             --shots 0 \
#             --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#             --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
#             --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2automatic_id_dict.pkl" \
#             --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/automatic_id2image_name_dict.pkl"
#     fi
# done
# eval "$(conda shell.bash hook)"
# conda activate openflamingo_deepspeed

CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --model_family flamingo \
    --cross_attn_every_n_layers 1 \
    --dataset_resampled \
    --batch_size_i2id 64 \
    --train_num_samples_i2id 80000000 \
    --batch_size_t2id 64 \
    --train_num_samples_t2id 80000000 \
    --workers=4 \
    --deepspeed \
    --deepspeed_stage 3 \
    --run_name "./checkpoints/random" \
    --precision bf16 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
    --learning_rate 1e-4 \
    --lr_scheduler linear \
    --warmup_steps  500 \
    --i2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker30k_i2id5/{000000000..00000006}.tar" \
    --wandb_project Gen_Cross_Modal-Retrieval \



# CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/eval/evaluate.py \
#     --model_family flamingo \
#     --vision_encoder_path ViT-L-14 \
#     --vision_encoder_pretrained openai\
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style  \
#     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style  \
#     --cross_attn_every_n_layers 1 \
#     --results_file results.json \
#     --precision fp32 \
#     --batch_size 1 \
#     --eval_flickr30 \
#     --shots 0 \
#     --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
#     --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
#     --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
#     --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/image_name2structured_id_dict.pkl" \
#     --add_extra_id_tokens "../AutoregressiveImageRetrieval/data/Openflamingo_format/structured_id2image_name_dict.pkl"



