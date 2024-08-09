#!/bin/sh


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path /storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/checkpoint_2.pt \
    --results_file results.json \
    --precision fp32 \
    --batch_size 8 \
    --eval_flickr_i2t \
    --shots 0 \
    --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
    --decoder_trie_path "../data/Openflamingo_format/flicker/caption_trie_test_set.pkl" \
    --constrained_num_beams 10 \
    --num_return_sequences 10

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path /storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/checkpoint_2.pt \
    --results_file results.json \
    --precision fp32 \
    --batch_size 8 \
    --eval_flickr_i2t \
    --shots 0 \
    --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
    --decoder_trie_path "../data/Openflamingo_format/flicker/caption_trie_test_set.pkl" \
    --constrained_num_beams 20 \
    --num_return_sequences 10

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path /storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/checkpoint_2.pt \
    --results_file results.json \
    --precision fp32 \
    --batch_size 8 \
    --eval_flickr_i2t \
    --shots 0 \
    --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
    --decoder_trie_path "../data/Openflamingo_format/flicker/caption_trie_test_set.pkl" \
    --constrained_num_beams 30 \
    --num_return_sequences 10

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path /storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/checkpoint_2.pt \
    --results_file results.json \
    --precision fp32 \
    --batch_size 8 \
    --eval_flickr_i2t \
    --shots 0 \
    --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
    --decoder_trie_path "../data/Openflamingo_format/flicker/caption_trie_test_set.pkl" \
    --constrained_num_beams 40 \
    --num_return_sequences 10


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path /storage_fast/yqli/project/AutoregressiveImageRetrieval/data/checkpoints/flicker30k_i2t_2/checkpoint_2.pt \
    --results_file results.json \
    --precision fp32 \
    --batch_size 8 \
    --eval_flickr_i2t \
    --shots 0 \
    --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "../data/dataset_flickr30k_coco_style.json" \
    --decoder_trie_path "../data/Openflamingo_format/flicker/caption_trie_test_set.pkl" \
    --constrained_num_beams 50 \
    --num_return_sequences 10
# ######random
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 ./open_flamingo/open_flamingo/train/finetuning.py \
  --lm_path anas-awadalla/mpt-1b-redpajama-200b \
  --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
  --cross_attn_every_n_layers 1 \
  --dataset_resampled \
  --batch_size_mmc4 64 \
  --train_num_samples_mmc4 1000000000000000 \
  --workers=4 \
  --run_name ../data/checkpoints/random \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --num_epochs 1 \
  --warmup_steps  100 \
  --mmc4_textsim_threshold 0.01 \
  --mmc4_shards "../data/Openflamingo_format/coco/coco_i2t/{000000000..000000082}.tar" \
  --logging_steps 1 \
  --mmc4_max_num_images 1 \
  --precision bf16 \
  --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
  --gradient_checkpointing \
  --fsdp \
  --fsdp_use_orig_params \
  --unfreeze_all