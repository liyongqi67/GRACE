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
```commandline

```
## Data
Our experiments are conducted on public Flickr30k and MS-COCO datasets, that produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from their original sources [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).
