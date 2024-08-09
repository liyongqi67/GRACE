from torch import nn
from .helpers import PerceiverResampler, GatedCrossAttentionBlock
from .vlm import VLMWithCrossAttention


class Flamingo(VLMWithCrossAttention):
    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_model: nn.Module,
        vis_feature_dim: int,
        initial_tokenizer_len: int,
        pad_token_id: int,
        cross_attn_every_n_layers: int = 1,
        decoder_layers_attr_name: str = None,
        gradient_checkpointing: bool = False,
        new_class_embed: bool = False,
    ):
        """
        Args:
            vision_encoder (nn.Module): HF CLIPModel
            lang_model (nn.Module): HF causal language model
            vis_feature_dim (int): final dimension of the visual features outputted by the vision_encoder
            initial_tokenizer_len (int): size of the tokenizer vocab
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
            decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
            gradient_checkpointing (bool, optional): whether to use gradient checkpointing. Defaults to False.
        """
        self._special_tokens = {
            "eoc_token": "<|endofchunk|>",
            "media_token": "<image>",
        }
        super().__init__(
            vision_encoder=vision_encoder,
            vision_tokenizer=PerceiverResampler(dim=vis_feature_dim),
            lang_model=lang_model,
            gradient_checkpointing=gradient_checkpointing,
            initial_tokenizer_len=initial_tokenizer_len,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            decoder_layers_attr_name=decoder_layers_attr_name,
            pad_token_id=pad_token_id,
        )
        if new_class_embed:
            import numpy as np
            import torch
            self.transform_layer = nn.Linear(2048, 768)
            image_emb = np.load('/storage_fast/yqli/project/AutoregressiveImageRetrieval/data/Openflamingo_format/coco/image_emb.npy')
            self.class_layer = nn.Linear(image_emb.shape[1], image_emb.shape[0], False)
            self.class_layer.weight = nn.Parameter(torch.tensor(image_emb.transpose(0, 1)))

    def set_trainable(self):
        """
        Freeze everything except: perceiver, gated_cross_attn_layers, and inserted LM input embeddings
        """
        self.requires_grad_(False)
        self.vision_tokenizer.requires_grad_(True)
        self.lang_model.requires_grad_(True)

        try:
            if getattr(self, 'class_layer'):
                print('will not train the class layer')
                self.class_layer.requires_grad_(False)
        except:
            print("An exception occurred in set_trainable of flamingo.py")
        # self.lang_model.gated_cross_attn_layers.requires_grad_(True)
        # self.lang_model.get_output_embeddings().set_requires_grad(
        #     require_regular_grad=False,
        #     require_additional_grad=True,
        # )
        # self.lang_model.get_input_embeddings().set_requires_grad(
        #     require_regular_grad=False,
        #     require_additional_grad=True,
        # )

    def _should_apply_weight_decay(self, parameter_name):
        """
        Flamingo applies 0.1 weight decay to cross attention parameters
        """
        return "gated_cross_attn" in parameter_name
