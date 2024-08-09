# Introduction
We have published several works on generative retrieval as follows.
```
Multiview Identifiers Enhanced Generative Retrieval. ACL 2023. (MINDER)
Generative Retrieval for Conversational Question Answering. IPM 2023. (GCoQA)
Learning to Rank in Generative Retrieval. AAAI 2024. (LTRGR)
Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond. ACL 2024 (GRACE).
Distillation Enhanced Generative Retrieval. ACL 2024 findings (DGR).
```
All code, data, and checkpoints of the above works are open-released:  
1. MINDER, LTRGR, and DGR, are a series of works on text retrieval. LTRGR and DGR are continuously training based on the MINDER model, so we release MINDER, LTRGR, and DGR together in the same repository https://github.com/liyongqi67/MINDER.  
2. GCoQA is the work on conversational retrieval and is released at https://github.com/liyongqi67/GCoQA.  
3. GRACE is the work on cross-modal retrieval and I am organizing the code.
# GRACE
This is the official implementation for the paper "Generative Cross-Modal Retrieval: Memorizing Images in Multimodal Language Models for Retrieval and Beyond".  
The preprint version is released in [Arxiv](Acknowledgments).  
If you find our paper or code helpful, please consider citing as follows:
```bibtex
@inproceedings{li-etal-2023-multiview,
    title = "Multiview Identifiers Enhanced Generative Retrieval",
    author = "Li, Yongqi  and Yang, Nan  and Wang, Liang  and Wei, Furu  and Li, Wenjie",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    pages = "6636--6648",
}
```
## Description
Our work is based on the Open-Flamingo project.   
However, we encountered some bugs when applying the FSDP training framework within Open-Flamingo (the 2023 version).   
As a result, we created two separate Open-Flamingo files: one for training (using our implemented DeepSpeed training framework) and one for inference.  
We use Conda to switch between the two Open-Flamingo environments.

## Install
Create the conda environment for openflamingo inference:
```commandline
cd open_flamingo
conda env create -f environment.yml
```
Create the conda environment for openflamingo deepspeed training:
```commandline
cd open_flamingo_deepspeed
conda env create -f environment.yml
```
## Dataset
Our experiments are conducted on public Flickr30k and MS-COCO datasets, that produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from their original sources [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).  The downloaded data is expected to be organized into the ./data/ directory as follows:  
├── dataset_coco.json  
├── dataset_flickr8k.json  
├── dataset_flickr30k.json  
├── dataset_flickr30k_coco_style.json  
├── Flickr30K/ # directory for Flickr images  
├── CoCo/ # directory for coco images  
├── Openflamingo_format/ #directory for processed files  
————————————————  
All the training data is processed from the original data and stored into ./data/Openflamingo_format/.
## Data process, training, and inference
### For numeric identifiers in the Flickr dataset
Generate the image-to-identifier (learning to memorize) data:
```bash
python ./data_process/convert_flicker30k_to_wds_i2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_i2numeric_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type numeric_identifier
```
Generate the query-to-identifier (learning to retrieve) data:
```bash
python ./data_process/convert_flicker30k_to_wds_t2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_t2numeric_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type numeric_identifier --pseudo_query ../data/Openflamingo_format/flicker/pseudo_query.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl
```
Generate a trie dictionary of identifiers for images in the test set to use for constrained generation.
```bash
python get_trie_dict_4structured_id.py --output_dir "../data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl" --json_file ../data/dataset_flickr30k.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl --identifier_type numeric_identifier
```
Training with the openflamingo deepspeed environment.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo_deepspeed/train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --model_family flamingo \
    --cross_attn_every_n_layers 1 \
    --dataset_resampled \
    --batch_size_i2id 16 \
    --train_num_samples_i2id 200000 \
    --batch_size_t2id 48 \
    --train_num_samples_t2id 600000 \
    --workers=4 \
    --deepspeed \
    --deepspeed_stage 3 \
    --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" \
    --precision bf16 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
    --learning_rate 1e-4 \
    --lr_scheduler linear \
    --warmup_steps  500 \
    --i2id_shards "./data/Openflamingo_format/flicker/flicker30k_i2numeric_id/{000000000..00000006}.tar" \
    --t2id_shards "./data/Openflamingo_format/flicker/flicker30k_t2numeric_id/{000000000..000000030}.tar" \
    --wandb_project Gen_Cross_Modal-Retrieval \
    --delete_previous_checkpoint \
    --report_to_wandb
```
Inference with the openflamingo environment.
```bash
eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id); do
    if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_numeric_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
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
            --flickr_image_dir_path "./data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "./data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "./data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "./data/Openflamingo_format/flicker/image_name2numeric_id_dict.pkl" \
            --id2image_name_dict "./data/Openflamingo_format/flicker/numeric_id2image_name_dict.pkl" \
            --decoder_trie_path "./data/Openflamingo_format/flicker/numeric_id_trie_test_set.pkl"
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed
```
### For string identifiers in the Flickr dataset
Generate the image-to-identifier (learning to memorize) data:
```bash
python ./data_process/convert_flicker30k_to_wds_i2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_i2string_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type string_identifier
```
Generate the query-to-identifier (learning to retrieve) data:
```bash
python ./data_process/convert_flicker30k_to_wds_t2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_t2string_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type string_identifier --pseudo_query ../data/Openflamingo_format/flicker/pseudo_query.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2string_id_dict.pkl
```
Generate a trie dictionary of identifiers for images in the test set to use for constrained generation.
```bash
python get_trie_dict_4structured_id.py --output_dir "../
data/Openflamingo_format/flicker/string_id_trie_test_set.pkl" --json_file ../data/dataset_flickr30k.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2string_id_dict.pkl --identifier_type string_identifier
```
Training with the openflamingo deepspeed environment.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo_deepspeed/train/train.py \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
    --model_family flamingo \
    --cross_attn_every_n_layers 1 \
    --dataset_resampled \
    --batch_size_i2id 16 \
    --train_num_samples_i2id 200000 \
    --batch_size_t2id 48 \
    --train_num_samples_t2id 600000 \
    --workers=4 \
    --deepspeed \
    --deepspeed_stage 3 \
    --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_string_id" \
    --precision bf16 \
    --num_epochs 5 \
    --gradient_checkpointing \
    --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
    --learning_rate 1e-4 \
    --lr_scheduler linear \
    --warmup_steps  500 \
    --i2id_shards "./data/Openflamingo_format/flicker/flicker30k_i2string_id/{000000000..00000006}.tar" \
    --t2id_shards ".data/Openflamingo_format/flicker/flicker30k_t2string_id/{000000000..000000030}.tar" \
    --wandb_project Gen_Cross_Modal-Retrieval \
    --delete_previous_checkpoint \
    --report_to_wandb
```
Inference with the openflamingo environment.
```bash
eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_string_id); do
    if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_string_id" ]; then
        echo $file
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
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
            --flickr_image_dir_path "./data/Flickr30K/flickr30k-images" \
            --flickr_karpathy_json_path "./data/dataset_flickr30k.json" \
            --flickr_annotations_json_path "./data/dataset_flickr30k_coco_style.json" \
            --image_name2id_dict "./data/Openflamingo_format/flicker/image_name2string_id_dict.pkl" \
            --id2image_name_dict "./data/Openflamingo_format/flicker/string_id2image_name_dict.pkl" \
            --decoder_trie_path "./data/Openflamingo_format/flicker/string_id_trie_test_set.pkl"
    fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed
```
### For semantic identifiers in the Flickr dataset
Generate the image-to-identifier (learning to memorize) data:
```bash
python ./data_process/convert_flicker30k_to_wds_i2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_i2semantic_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type semantic_identifier --pseudo_query ../data/Openflamingo_format/flicker/pseudo_query.json
```
Generate the query-to-identifier (learning to retrieve) data:
```bash
python ./data_process/convert_flicker30k_to_wds_t2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_t2semantic_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type semantic_identifier --pseudo_query ../data/Openflamingo_format/flicker/pseudo_query.json --image_name2id_dict ../data/Openflamingo_format/flicker/flicker30k_i2semantic_id/image_name2semantic_id_dict.pkl
```
Generate a trie dictionary of identifiers for images in the test set to use for constrained generation.
```bash
python get_trie_dict_4structured_id.py --output_dir "../data/Openflamingo_format/flicker/semantic_id_trie_test_set.pkl" --json_file ../data/dataset_flickr30k.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2semantic_id_dict.pkl --identifier_type semantic_identifier
```
Training with the openflamingo deepspeed environment.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo_deepspeed/train/train.py \
     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
     --model_family flamingo \
     --cross_attn_every_n_layers 1 \
     --dataset_resampled \
     --batch_size_i2id 16 \
     --train_num_samples_i2id 200000 \
     --batch_size_t2id 48 \
     --train_num_samples_t2id 600000 \
     --workers=4 \
     --deepspeed \
     --deepspeed_stage 3 \
     --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_semantic_id" \
     --precision bf16 \
     --num_epochs 5 \
     --gradient_checkpointing \
     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
     --learning_rate 1e-4 \
     --lr_scheduler linear \
     --warmup_steps  500 \
     --i2id_shards "./data/Openflamingo_format/flicker/flicker30k_i2semantic_id/{000000000..00000006}.tar" \
     --t2id_shards "./data/Openflamingo_format/flicker/flicker30k_t2semantic_id/{000000000..000000029}.tar" \
     --wandb_project Gen_Cross_Modal-Retrieval \
     --delete_previous_checkpoint \
     --report_to_wandb
```
Inference with the openflamingo environment.
```bash
eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_semantic_id); do
     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_semantic_id" ]; then
         echo $file
         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
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
             --flickr_image_dir_path "./data/Flickr30K/flickr30k-images" \
             --flickr_karpathy_json_path "./data/dataset_flickr30k.json" \
             --flickr_annotations_json_path "./data/dataset_flickr30k_coco_style.json" \
             --image_name2id_dict "./data/Openflamingo_format/flicker/image_name2semantic_id_dict.pkl" \
             --id2image_name_dict "./data/Openflamingo_format/flicker/semantic_id2image_name_dict.pkl" \
             --decoder_trie_path "./data/Openflamingo_format/flicker/semantic_id_trie_test_set.pkl"
     fi
 done
 eval "$(conda shell.bash hook)"
 conda activate openflamingo_deepspeed
```
### For structured identifiers in the Flickr dataset
Generate the structured identifiers:
```bash
python data_process/structured_id.py --dataset ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images
```
Generate the image-to-identifier (learning to memorize) data:
```bash
python ./data_process/convert_flicker30k_to_wds_i2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_i2structured_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type structured_identifier --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2structured_id_dict.pkl
```
Generate the query-to-identifier (learning to retrieve) data:
```bash
python ./data_process/convert_flicker30k_to_wds_t2id7.py --output_dir ../data/Openflamingo_format/flicker/flicker30k_t2structured_id --json_file ../data/dataset_flickr30k.json --image_dir ../data/Flickr30K/flickr30k-images --identifier_type structured_identifier --pseudo_query ../data/Openflamingo_format/flicker/pseudo_query.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2structured_id_dict.pkl
```
Generate a trie dictionary of identifiers for images in the test set to use for constrained generation.
```bash
python get_trie_dict_4structured_id.py --output_dir "../data/Openflamingo_format/flicker/structured_id_trie_test_set.pkl" --json_file ../data/dataset_flickr30k.json --image_name2id_dict ../data/Openflamingo_format/flicker/image_name2structured_id_dict.pkl --id2image_name_dict ../data/Openflamingo_format/flicker/structured_id2image_name_dict.pkl --identifier_type structured_identifier
```
Training with the openflamingo deepspeed environment.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run --nnodes=1 --nproc_per_node=4 ./open_flamingo_deepspeed/train/train.py \
     --lm_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
     --tokenizer_path anas-awadalla/mpt-1b-redpajama-200b-hf-style \
     --model_family flamingo \
     --cross_attn_every_n_layers 1 \
     --dataset_resampled \
     --batch_size_i2id 16 \
     --train_num_samples_i2id 200000 \
     --batch_size_t2id 48 \
     --train_num_samples_t2id 600000 \
     --workers=4 \
     --deepspeed \
     --deepspeed_stage 3 \
     --run_name "./checkpoints/deepspeed3_bf16_i2id_t2id_structured_id" \
     --precision bf16 \
     --num_epochs 5 \
     --gradient_checkpointing \
     --pretrained_checkpoint openflamingo/OpenFlamingo-3B-vitl-mpt1b \
     --learning_rate 1e-4 \
     --lr_scheduler linear \
     --warmup_steps  500 \
     --add_extra_id_tokens "./data/Openflamingo_format/flicker/structured_id2image_name_dict.pkl" \
     --i2id_shards "./data/Openflamingo_format/flicker/flicker30k_i2structured_id/{000000000..00000006}.tar" \
     --t2id_shards "./data/Openflamingo_format/flicker/flicker30k_t2structured_id/{000000000..000000030}.tar" \
     --wandb_project Gen_Cross_Modal-Retrieval \
     --delete_previous_checkpoint \
     --report_to_wandb
```
Inference with the openflamingo environment.
```bash
eval "$(conda shell.bash hook)"
conda activate openflamingo
for file in $(find ./checkpoints/deepspeed3_bf16_i2id_t2id_structured_id); do
     if [ -d "$file" ] && [ "$file" != "./checkpoints/deepspeed3_bf16_i2id_t2id_structured_id" ]; then
         echo $file
         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=1997 ./open_flamingo/open_flamingo/eval/evaluate.py \
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
             --flickr_image_dir_path "./data/Flickr30K/flickr30k-images" \
             --flickr_karpathy_json_path "../AutoregressiveImageRetrieval/data/dataset_flickr30k.json" \
             --flickr_annotations_json_path "./data/dataset_flickr30k_coco_style.json" \
             --image_name2id_dict "./data/Openflamingo_format/flicker/image_name2structured_id_dict.pkl" \
             --id2image_name_dict "./data/Openflamingo_format/flicker/structured_id2image_name_dict.pkl" \
             --add_extra_id_tokens "./data/Openflamingo_format/flicker/structured_id2image_name_dict.pkl" \
             --decoder_trie_path "./data/Openflamingo_format/flicker/structured_id_trie_test_set.pkl"
     fi
done
eval "$(conda shell.bash hook)"
conda activate openflamingo_deepspeed
```
