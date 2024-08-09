from transformers import AutoModelForCausalLM, AutoTokenizer
import open_clip

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance
from transformers import AutoConfig
import pickle
from peft import LoraConfig, get_peft_model
import torch.nn as nn
def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    unfreeze_all: bool = False,
    add_extra_id_tokens: str = None,
    lora: bool = False,
    dropout: bool = False,
    new_class_embed: bool = False,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if add_extra_id_tokens != None:
        with open(add_extra_id_tokens, 'rb') as f:
            add_extra_id_tokens = pickle.load(f)
        id_token_dict = {}    
        for key in add_extra_id_tokens:
            for s in key.split("-"):
                id_token_dict[s]=1
        text_tokenizer.add_tokens(list(id_token_dict.keys()))

    config = AutoConfig.from_pretrained(lang_encoder_path, trust_remote_code=True,)
    if dropout:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
            lang_encoder_path,
            local_files_only=use_local_files,
            trust_remote_code=True,
            attn_pdrop = 0.1,
            emb_pdrop = 0.1,
            resid_pdrop = 0.1
        )
    else:
        lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings                
            def get_output_embeddings(self): #yonqgi add, but it seems wrong
               return self.transformer.wte
            def set_output_embeddings(self, new_embeddings):
              self.transformer.wte = new_embeddings
        extend_instance(lang_encoder, EmbeddingFnMixin)


    # convert LM to FlamingoLM
    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    # peft_config = LoraConfig(
    # task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules = ["Wqkv"])
    # lang_encoder = get_peft_model(lang_encoder, peft_config)

    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        new_class_embed = new_class_embed,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    if lora:
        model = prepare_model_for_tuning(model)
        print(
            f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
    else:
        # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
        model.perceiver.requires_grad_(True)
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        if not freeze_lm_embeddings:
            model.lang_encoder.get_input_embeddings().requires_grad_(True)
            # model.requires_grad_(True)
            # TODO: investigate also training the output embeddings when untied
        if unfreeze_all:
            model.requires_grad_(True)
            model.vision_encoder.requires_grad_(False)
        print(
            f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
    return model, image_processor, text_tokenizer
def prepare_model_for_tuning(model: nn.Module):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["Wqkv"],
        lora_dropout=0.1,
        bias="none",  # won't use bias currently
        modules_to_save=[],  # TODO: might be helpful if save partial model
        task_type="VL",
    )
    model.lang_encoder = get_peft_model(model.lang_encoder, peft_config=lora_config)
    return model


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
}
