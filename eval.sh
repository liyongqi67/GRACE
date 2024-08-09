#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1

for file in $(find ../data/checkpoints/flicker30k_i2t_6epoch/ -name "*.pt"); do
    echo $file
    CUDA_VISIBLE_DEVICES=6  torchrun --nnodes=1 --nproc_per_node=1 ./open_flamingo/open_flamingo/eval/evaluate.py \
        --vision_encoder_path ViT-L-14 \
        --vision_encoder_pretrained openai\
        --lm_path anas-awadalla/mpt-1b-redpajama-200b \
        --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
        --cross_attn_every_n_layers 1 \
        --checkpoint_path $file \
        --results_file "results.json" \
        --precision amp_bf16 \
        --batch_size 8 \
        --eval_flickr_i2t \
        --shots 0 \
        --flickr_image_dir_path "../data/Flickr30K/flickr30k-images" \
        --flickr_karpathy_json_path "../data/dataset_flickr30k.json" \
        --flickr_annotations_json_path "../data/dataset_flickr30k.json" \
        --decoder_trie_path "../data/Openflamingo_format/caption_trie_test_set.pkl"
done


