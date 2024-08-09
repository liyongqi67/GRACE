#!/bin/sh






eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find /ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id); do
    if [ -d "$file" ] && [ "$file" != "/ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
            --vision_encoder_path ViT-L-14 \
            --vision_encoder_pretrained openai\
            --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --cross_attn_every_n_layers 1 \
            --checkpoint_path $file \
            --results_file results.json \
            --precision fp32 \
            --batch_size 8 \
            --eval_flickr_t2id \
            --shots 0 \
            --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl" \
            --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id2image_name_dict.pkl" \
            --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl" \
            --constrained_num_beams 20
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed


eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find /ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id); do
    if [ -d "$file" ] && [ "$file" != "/ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
            --vision_encoder_path ViT-L-14 \
            --vision_encoder_pretrained openai\
            --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --cross_attn_every_n_layers 1 \
            --checkpoint_path $file \
            --results_file results.json \
            --precision fp32 \
            --batch_size 8 \
            --eval_flickr_t2id \
            --shots 0 \
            --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl" \
            --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id2image_name_dict.pkl" \
            --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl" \
            --constrained_num_beams 30
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed


eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find /ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id); do
    if [ -d "$file" ] && [ "$file" != "/ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
            --vision_encoder_path ViT-L-14 \
            --vision_encoder_pretrained openai\
            --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --cross_attn_every_n_layers 1 \
            --checkpoint_path $file \
            --results_file results.json \
            --precision fp32 \
            --batch_size 8 \
            --eval_flickr_t2id \
            --shots 0 \
            --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl" \
            --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id2image_name_dict.pkl" \
            --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl" \
            --constrained_num_beams 40
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed



eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find /ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id); do
    if [ -d "$file" ] && [ "$file" != "/ssd4/yqli/checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ../AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/eval/evaluate.py \
            --vision_encoder_path ViT-L-14 \
            --vision_encoder_pretrained openai\
            --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
            --cross_attn_every_n_layers 1 \
            --checkpoint_path $file \
            --results_file results.json \
            --precision fp32 \
            --batch_size 8 \
            --eval_flickr_t2id \
            --shots 0 \
            --flickr_image_dir_path "../AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "../AutoregressiveImageRetrieval//data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl" \
            --id2image_name_dict "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id2image_name_dict.pkl" \
            --decoder_trie_path "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl" \
            --constrained_num_beams 50
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed









# ######random
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 /storage_fast/yqli/project/AutoregressiveImageRetrieval/code/open_flamingo/open_flamingo/train/finetuning.py \
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
  --mmc4_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/coco/coco_i2t/{000000000..000000082}.tar" \
  --logging_steps 1 \
  --mmc4_max_num_images 1 \
  --precision bf16 \
  --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
  --gradient_checkpointing \
  --fsdp \
  --fsdp_use_orig_params \
  --unfreeze_all





CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo/train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --model_family flamingo \
    --cross_attn_every_n_layers 1 \
    --dataset_resampled \
    --batch_size_t2id 64 \
    --train_num_samples_t2id 600000000000 \
    --workers=4 \
    --deepspeed \
    --deepspeed_stage 3 \
    --run_name "./checkpoints/random" \
    --precision bf16 \
    --num_epochs 1 \
    --gradient_checkpointing \
    --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
    --learning_rate 1e-4 \
    --lr_scheduler linear \
    --warmup_steps  500 \
    --t2id_shards "../AutoregressiveImageRetrieval/data/Openflamingo_format/flicker/flicker30k_t2automatic_id/{000000000..000000030}.tar" \
    --new_class_embed \
    --loss Classifier_loss \
    --wandb_project Gen_Cross_Modal-Retrieval \
    --delete_previous_checkpoint \








