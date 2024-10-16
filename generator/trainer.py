import inspect
import warnings
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

class GENTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        loss_type: Literal["sigmoid", "hinge"] = "sigmoid",
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the trainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError("")
        
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the trainer. This will automatically create models for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the trainer. This will automatically create a ref model for you."
            )
        
        if not is_peft_available() and peft_config is not None:
            raise ValueError("PEFT is not installed and you passed a 'peft_config' to the trainer, please install it to use the PEFT models.")
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
            
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )
                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}
                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs
                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(model, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            model = get_peft_model(model, peft_config)
        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        if generate_during_eval and not is_wandb_available():
            raise ValueError("You passed 'generate_during_eval=True' to the trainer, please install wandb to use the generation during eval.")
        
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("Please pass is_encoder_decoder to the trainer.")
        else:
            self.is_encoder_decoder = is_encoder_decoder
        
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)
        
        if data_collator is None:
            if tokenizer is None:
                raise ValueError("max_length or a tokenizer must be specified when using the default.")
            if max_length is None:
                warnings.warn(
                    "You should set 'max_length' in the trainer",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using the default data_collator, you should set 'max_prompt_length' in the trainer",
                    UserWarning
                )
                max_prompt_length = 128
            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "You should set 'max_target_length' in the trainer when using the default data_collator",
                    UserWarning,
                )
                max_target_length = 128
            
            data_collator = DataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
                is_encoder_decoder=self.is_encoder_decoder,
                max_target_length=max_target_length,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "You should set 'remove_unused_columns' to False when using the default data_collator",
                    UserWarning
                )
                self.use_data_collator = True
            else:
                self.use_data_collator = False
            
            if disable_dropout:
                disable_dropout_in_model(model)
                if self.ref_model is not None:
                    disable_dropout_in_model(self.ref_model)
            
            self.max_length = max_length
            self.generate_during_eval = generate_during_eval
            self.label_pad_token_id = label_pad_token_id
            self.padding_value = padding_value

            self.beta = beta
            self.loss_type = loss_type

            self._stored_metrics = defaultdict(lambda: defaultdict(list))
            
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                model_init=model_init,
                compute_metrics=compute_metrics,
                callbacks=callbacks,
                optimizers=optimizers,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )

            if not hasattr(self, "accelerator"):
                raise AttributeError(
                    "Your traininger does not have an accelerate object. Consider upgrading 'transformers'."
                )
        
            if self.ref_model is None:
                if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapter"):
                    raise ValueError(
                        "You are using a 'peft' version that does not support the 'disable_adapter'. Please update your 'peft' version to the latest version."
                    )
            else:
                if self.is_deepspeed_enabled:
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
                else:
                    self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        
        def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
            if model is not None:
                if hasattr(model, "config"):
                    hidden_size = (
                        max(model.config.hidden_sizes)
                        if getattr(model.config, "hidden_sizes", None)
                        else getattr(model.config, "hidden_size", None)
                    )
                    if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                        config_kwargs.update(
                            {
                                "zero_otimization.reduce_bucket_size": hidden_size,
                                "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                                "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                            }
                        )
            if config_kwargs["zero_optimization"]["stage"] != 3:
                config_kwargs["zero_optimization"]["stage"] = 0
            model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
            model.eval()
            return model

        def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
            concatenated_batch = {}
            if self.is_encoder_decoder:
                max_length = max(batch["real_labels"].shape[1], batch["generated_labels"].shape[1])
            else:
                max_legnth = max(batch["real_input_ids"].shape[1], batch["generated_input_ids"].shape[1])
            
            for k in batch:
                if k.startswith("real") and isinstance(batch[k], torch.Tensor):
                    pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                    concatenated_key = k.replace("generated", "concatenated")
                    concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
