#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

CUDA_VISIBLE_DEVICES=6,7 python open_flamingo/open_flamingo/eval/evaluate.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai\
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --cross_attn_every_n_layers 1 \
    --checkpoint_path "/home/share/yongqi/project/AutoregressiveImageRetrieval/code/OpenFlamingo-3B-vitl-mpt1b/checkpoint_0.pt" \
    --results_file "results.json" \
    --precision amp_bf16 \
    --batch_size 8 \
    --eval_flickr30 \
    --flickr_image_dir_path "/home/share/yongqi/project/AutoregressiveImageRetrieval/data/Flickr30K/flickr30k-images" \
    --flickr_karpathy_json_path "/home/share/yongqi/project/AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
    --flickr_annotations_json_path "/home/share/yongqi/project/AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \




# srun --cpu_bind=v --accel-bind=gn python open_flamingo/open_flamingo/eval/evaluate.py \
#     --vision_encoder_path ViT-L-14 \
#     --vision_encoder_pretrained openai\
#     --lm_path anas-awadalla/mpt-1b-redpajama-200b \
#     --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
#     --cross_attn_every_n_layers 1 \
#     --checkpoint_path "openflamingo/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt" \
#     --results_file "results.json" \
#     --precision amp_bf16 \
#     --batch_size 8 \
#     --eval_coco \
#     --eval_vqav2 \
#     --eval_flickr30 \
#     --eval_ok_vqa \
#     --eval_textvqa \
#     --eval_vizwiz \
#     --eval_hateful_memes \
#     --coco_train_image_dir_path "/path/to/mscoco_karpathy/train2014" \
#     --coco_val_image_dir_path "/path/to/mscoco_karpathy/val2014" \
#     --coco_karpathy_json_path "/path/to/mscoco_karpathy/dataset_coco.json" \
#     --coco_annotations_json_path "/path/to/mscoco_karpathy/annotations/captions_val2014.json" \
#     --vqav2_train_image_dir_path "/path/to/vqav2/train2014" \
#     --vqav2_train_annotations_json_path "/path/to/vqav2/v2_mscoco_train2014_annotations.json" \
#     --vqav2_train_questions_json_path "/path/to/vqav2/v2_OpenEnded_mscoco_train2014_questions.json" \
#     --vqav2_test_image_dir_path "/path/to/vqav2/val2014" \
#     --vqav2_test_annotations_json_path "/path/to/vqav2/v2_mscoco_val2014_annotations.json" \
#     --vqav2_test_questions_json_path "/path/to/vqav2/v2_OpenEnded_mscoco_val2014_questions.json" \
#     --flickr_image_dir_path "/path/to/flickr30k/flickr30k-images" \
#     --flickr_karpathy_json_path "/path/to/flickr30k/dataset_flickr30k.json" \
#     --flickr_annotations_json_path "/path/to/flickr30k/dataset_flickr30k_coco_style.json" \
#     --ok_vqa_train_image_dir_path "/path/to/okvqa/train2014" \
#     --ok_vqa_train_annotations_json_path "/path/to/okvqa/mscoco_train2014_annotations.json" \
#     --ok_vqa_train_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_train2014_questions.json" \
#     --ok_vqa_test_image_dir_path "/path/to/okvqa/val2014" \
#     --ok_vqa_test_annotations_json_path "/path/to/okvqa/mscoco_val2014_annotations.json" \
#     --ok_vqa_test_questions_json_path "/path/to/okvqa/OpenEnded_mscoco_val2014_questions.json" \
#     --textvqa_image_dir_path "/path/to/textvqa/train_images/" \
#     --textvqa_train_questions_json_path "/path/to/textvqa/train_questions_vqa_format.json" \
#     --textvqa_train_annotations_json_path "/path/to/textvqa/train_annotations_vqa_format.json" \
#     --textvqa_test_questions_json_path "/path/to/textvqa/val_questions_vqa_format.json" \
#     --textvqa_test_annotations_json_path "/path/to/textvqa/val_annotations_vqa_format.json" \
#     --vizwiz_train_image_dir_path "/path/to/v7w/train" \
#     --vizwiz_test_image_dir_path "/path/to/v7w/val" \
#     --vizwiz_train_questions_json_path "/path/to/v7w/train_questions_vqa_format.json" \
#     --vizwiz_train_annotations_json_path "/path/to/v7w/train_annotations_vqa_format.json" \
#     --vizwiz_test_questions_json_path "/path/to/v7w/val_questions_vqa_format.json" \
#     --vizwiz_test_annotations_json_path "/path/to/v7w/val_annotations_vqa_format.json" \
#     --hateful_memes_image_dir_path "/path/to/hateful_memes/img" \
#     --hateful_memes_train_annotations_json_path "/path/to/hateful_memes/train.json" \
#     --hateful_memes_test_annotations_json_path "/path/to/hateful_memes/dev.json" \