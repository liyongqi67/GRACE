from typing import List

from PIL import Image
import torch
torch.cuda.empty_cache()
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.src.factory import create_model_and_transforms
from contextlib import suppress
from open_flamingo.eval.models.utils import unwrap_model
from huggingface_hub import hf_hub_download

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
class EvalModel(BaseEvalModel):
    """OpenFlamingo model evaluation.

    Attributes:
      model (nn.Module): Underlying Torch model.
      tokenizer (transformers.PreTrainedTokenizer): Tokenizer for model.
      device: Index of GPU to use, or the string "CPU"
    """

    def __init__(self, model_args,decoder_trie=None, args=None):
        assert (
            "vision_encoder_path" in model_args
            and "lm_path" in model_args
            and "checkpoint_path" in model_args
            and "lm_tokenizer_path" in model_args
            and "cross_attn_every_n_layers" in model_args
            and "vision_encoder_pretrained" in model_args
            and "precision" in model_args
        ), "OpenFlamingo requires vision_encoder_path, lm_path, device, checkpoint_path, lm_tokenizer_path, cross_attn_every_n_layers, vision_encoder_pretrained, and precision arguments to be specified"

        self.device = (
            model_args["device"]
            if ("device" in model_args and model_args["device"] >= 0)
            else "cpu"
        )
        if args!= None and args.add_extra_id_tokens!= None:
            (
                self.model,
                self.image_processor,
                self.tokenizer,
            ) = create_model_and_transforms(
                model_args["vision_encoder_path"],
                model_args["vision_encoder_pretrained"],
                model_args["lm_path"],
                model_args["lm_tokenizer_path"],
                cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
                add_extra_id_tokens = args.add_extra_id_tokens,
                lora = args.lora,
                new_class_embed = args.new_class_embed,
            )  
        else:
            (
                self.model,
                self.image_processor,
                self.tokenizer,
            ) = create_model_and_transforms(
                model_args["vision_encoder_path"],
                model_args["vision_encoder_pretrained"],
                model_args["lm_path"],
                model_args["lm_tokenizer_path"],
                cross_attn_every_n_layers=int(model_args["cross_attn_every_n_layers"]),
                new_class_embed = args.new_class_embed,
            )

        if ".pt" in model_args["checkpoint_path"]:
            checkpoint = torch.load(model_args["checkpoint_path"], map_location=torch.device('cpu'))
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
                checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            if "module" in checkpoint:
                checkpoint = checkpoint["module"]
            checkpoint = {k.replace("_checkpoint_wrapped_module.", "").replace("lang_model", "lang_encoder").replace("vision_tokenizer", "perceiver"): v for k, v in checkpoint.items()}
        else:
            # checkpoint = get_fp32_state_dict_from_zero_checkpoint("/".join(model_args["checkpoint_path"].split("/")[:-1]),tag=model_args["checkpoint_path"].split("/")[-1])
            checkpoint = get_fp32_state_dict_from_zero_checkpoint("/".join(model_args["checkpoint_path"].split("/")[:-1]),tag=model_args["checkpoint_path"].split("/")[-1])
            checkpoint = {k.replace("_checkpoint_wrapped_module.", "").replace("lang_model", "lang_encoder").replace("vision_tokenizer", "perceiver"): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)

        self.model.eval()
        self.tokenizer.padding_side = "left"

        # autocast
        self.autocast = get_autocast(model_args["precision"])
        self.cast_dtype = get_cast_dtype(model_args["precision"])
        self.decoder_trie=decoder_trie
        self.prefix_len=0



    def _prepare_images(self, batch: List[List[torch.Tensor]]) -> torch.Tensor:
        """Preprocess images and stack them.

        Args:
            batch: A list of lists of images.

        Returns:
            A Tensor of shape
            (batch_size, images_per_example, frames, channels, height, width).
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)

                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed
        return batch_images


    def get_outputs(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        num_return_sequences = 1,
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                    input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    num_return_sequences=num_return_sequences
                )
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def get_outputs_classifier(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        with torch.inference_mode():
            with self.autocast():
                outputs = self.model(
                    vision_x=self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True),
                    lang_x= input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True),
                )
        last_layer_hidden_states = outputs.hidden_states[-1]
        last_hidden_state = last_layer_hidden_states[:, -1, :]
        forward_tensor = unwrap_model(self.model).transform_layer(last_hidden_state)
        forward_tensor = unwrap_model(self.model).class_layer(forward_tensor)

        return forward_tensor
    def prefix_allowed_tokens_fn(self, batch_id, sent):
        # print(sent.tolist())
        # print(self.tokenizer.decode(sent.tolist()))
        return self.decoder_trie.get(sent.tolist()[self.prefix_len-1:])

    def get_outputs_contrained(
        self,
        batch_text: List[str],
        batch_images: List[List[Image.Image]],
        min_generation_length: int,
        max_generation_length: int,
        num_beams: int,
        length_penalty: float,
        num_return_sequences = 1,
    ) -> List[str]:
        encodings = self.tokenizer(
            batch_text,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=2000,
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        self.prefix_len = len(input_ids[0])
        with torch.inference_mode():
            with self.autocast():
                outputs = unwrap_model(self.model).generate(
                    self._prepare_images(batch_images).to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                    input_ids.to(self.device, dtype=self.cast_dtype, non_blocking=True),
                    attention_mask=attention_mask.to(
                        self.device, dtype=self.cast_dtype, non_blocking=True
                    ),
                    min_new_tokens=min_generation_length,
                    max_new_tokens=max_generation_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                    num_return_sequences=num_return_sequences
                )
        # print(outputs)
        # print(self.tokenizer.batch_decode(outputs))
        outputs = outputs[:, len(input_ids[0]) :]

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)



    def get_logits(
        self,
        lang_x: torch.Tensor,
        vision_x: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        clear_conditioned_layers: bool = False,
    ):
        with torch.inference_mode():
            with self.autocast():
                outputs = self.model(
                    vision_x=vision_x,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                    clear_conditioned_layers=clear_conditioned_layers,
                    past_key_values=past_key_values,
                    use_cache=(past_key_values is not None),
                )
        return outputs

    def encode_vision_x(self, image_tensor: torch.Tensor):
        unwrap_model(self.model)._encode_vision_x(image_tensor.to(self.device))

    def uncache_media(self):
        unwrap_model(self.model).uncache_media()

    def cache_media(self, input_ids, vision_x):
        unwrap_model(self.model).cache_media(input_ids=input_ids, vision_x=vision_x)

    def get_vqa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def get_caption_prompt(self, caption=None) -> str:
        return f"<image>Output:{caption if caption is not None else ''}{'<|endofchunk|>' if caption is not None else ''}"

    def get_i2t_prompt(self, caption=None) -> str:
        return f"<image>"
    def get_i2id_prompt(self, caption=None) -> str:
        return f"image numeric id<image>"
    def get_t2id_prompt(self, caption=None) -> str:
        return f"caption:{' '.join(caption.split(' '))}<image>"
    def get_t2id_prompt_classifier(self, caption=None) -> str:
        return f"caption:{' '.join(caption.split(' '))}<image><|endofchunk|>"
    def get_id2caption_prompt(self, caption=None) -> str:
        return f"Describe the image id {caption}<image>"
def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        return suppress
