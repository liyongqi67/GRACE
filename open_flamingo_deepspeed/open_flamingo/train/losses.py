from open_flamingo.src.vlm import VLM
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
torch.set_printoptions(profile="full")
SUPPORTED_LOSSES = ["next_token_prediction", "next_token_prediction_with_z_loss","Classifier_loss"]


def get_loss_fn(loss_name):
    if loss_name == "next_token_prediction":
        return NextTokenPrediction()
    elif loss_name == "next_token_prediction_with_z_loss":
        return NextTokenPredictionWithZLoss()
    elif loss_name == "Classifier_loss":
        return Classifier_loss()
    else:
        raise ValueError(
            f"Loss {loss_name} not supported. Supported losses: {SUPPORTED_LOSSES}"
        )


class Loss:
    @property
    def name(self):
        raise NotImplementedError

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        """
        Args:
            model: VLM model
            images: images tensor, already moved to device and cast to appropriate dtype
                shape (B, T_img, F, C, H, W)
            input_ids: input ids tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            attention_mask: attention mask tensor, already moved to device and cast to appropriate dtype
                shape (B, T_text)
            autocast: autocast context manager
        Return:
            loss: scalar loss
        """
        raise NotImplementedError


class NextTokenPredictionWithZLoss(Loss):
    @property
    def name(self):
        return "next_token_prediction_with_z_loss"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
        z_loss_eps: float = 1e-4,
    ):
        # set up labels; language model is expected to handle shifting
        labels = input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.eos_token] = -100



        #######yongqi add for label mask
        media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)[
            "input_ids"
        ][-1]
        for i in range(labels.shape[0]):
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < labels.shape[1] and labels[i][label_idx] != media_token_id   ###########labels[i][label_idx] != media_token_id 改为input_ids[i][label_idx] != media_token_id     
            ):  
                labels[i][label_idx] = -100
                label_idx += 1
            # get index of all endofchunk tokens in the sequence
            endofchunk_idxs = torch.where(labels[i] == endofchunk_token_id)[0]
            for endofchunk_idx in endofchunk_idxs:
                token_idx = endofchunk_idx + 1
                while (
                    token_idx < labels.shape[1]
                    and labels[i][token_idx] != media_token_id
                ):
                    labels[i][token_idx] = -100
                    token_idx += 1
        #######yongqi add for label mask

        ###########yongqi mask for train <|endofchunk|>
        labels[labels == media_token_id] = -100
        # special_token_ids = torch.Tensor(unwrap_model(model).special_token_ids).to(
        #     labels.device
        # )
        # labels[torch.isin(labels, special_token_ids)] = -100
        labels = labels.to(input_ids.device)
        # if input_ids.get_device()==0:
        #     print("input_ids: ", tokenizer.batch_decode(input_ids))
            # print("input_ids: ", input_ids)     
        #     print("labels: ", labels)    
        # call forward
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

#####################
        # # To get the hidden states from the last layer
        # last_layer_hidden_states = output.hidden_states[-1]
        # last_hidden_state = last_layer_hidden_states[:, -1, :]
        # print(last_hidden_state.shape)

########################################################
        logits = output[1]

        logits = logits.float()

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLossWithZLoss(eps=z_loss_eps)
        shift_logits = shift_logits.view(-1, model.lang_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss





class NextTokenPrediction(NextTokenPredictionWithZLoss):
    # same as NextTokenPredictionWithZLoss, but with z_loss_eps = 0
    @property
    def name(self):
        return "next_token_prediction"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
    ):
        return super().__call__(
            model=model,
            tokenizer=tokenizer,
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            autocast=autocast,
            z_loss_eps=0,
        )


def unwrap_model(model):
    """
    Unwrap a model from a DataParallel or DistributedDataParallel wrapper.
    """
    if isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        return model.module
    else:
        return model


# From OpenLM (https://github.com/mlfoundations/open_lm/blob/main/open_lm/losses.py)
class CrossEntropyLossWithZLoss(CrossEntropyLoss):
    def __init__(
        self,
        eps: float = 1e-4,
        weight: Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        super().__init__(
            weight, size_average, ignore_index, reduce, reduction, label_smoothing
        )
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.eps == 0:
            return super().forward(input, target)

        return super().forward(input, target) + self.eps * torch.square(
            torch.logsumexp(input, dim=-1).mean()
        )
class Classifier_loss(Loss):
    @property
    def name(self):
        return "Classifier_loss"

    def __call__(
        self,
        model: VLM,
        tokenizer,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast: callable,
        target:torch.Tensor,
    ):
        # if input_ids.get_device()==0:
        #     print("input_ids: ", tokenizer.batch_decode(input_ids))
        with autocast():
            output = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
            )
        # To get the hidden states from the last layer
        last_layer_hidden_states = output.hidden_states[-1]
        last_hidden_state = last_layer_hidden_states[:, -1, :]
        forward_tensor = model.transform_layer(last_hidden_state)
        forward_tensor = model.class_layer(forward_tensor)
        loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(forward_tensor, target.view(-1))

        return loss

