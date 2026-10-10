import inspect
from typing import Any, Dict, Optional
import torch
from transformers import T5ForConditionalGeneration
from transformers.generation.configuration_utils import GenerationConfig
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from .block_mapping import injection_blocks_map

class ChronosX(T5ForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        T5ForConditionalGeneration.__init__(self, *args, **kwargs)

        self.output_hidden_states = False

        if self.input_injection_class:
            self.input_injection_block = self.input_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
            )
            self.input_injection_block_decoder = self.input_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
            )
        else:
            self.input_injection_block = None
            self.input_injection_block_decoder = None

        if self.output_injection_class:
            self.output_injection_block = self.output_injection_class(
                hidden_dim=self.hidden_dim,
                model_dim=self.model_dim,
                num_covariates=self.num_covariates,
                num_layers=self.num_layers,
                vocab_size=self.vocab_size,
            )
            self.output_hidden_states = True
        else:
            self.output_injection_block = None

    @classmethod
    def set_state(
        cls,
        num_covariates,
        covariate_injection,
        hidden_dim,
        num_layers,
        vocab_size,
        model_dim,
    ):

        cls.num_covariates = num_covariates
        cls.covariate_injection = covariate_injection
        cls.hidden_dim = hidden_dim
        cls.num_layers = num_layers
        cls.vocab_size = vocab_size
        cls.model_dim = model_dim

        if cls.covariate_injection in injection_blocks_map.keys():
            (cls.input_injection_class, cls.output_injection_class) = (
                injection_blocks_map[cls.covariate_injection]
            )
        else:
            cls.input_injection_class = None
            cls.output_injection_class = None

        return cls

    def initialize_blocks(self):
        if self.input_injection_block:
            self.input_injection_block.initialize_modules()
            self.input_injection_block_decoder.initialize_modules()

        if self.output_injection_block:
            self.output_injection_block.initialize_modules()

    def freeze(self, layer_name=None):
        for name, param in self.named_parameters():
            if layer_name == "all" or layer_name in name:
                param.requires_grad = False

    def unfreeze(self, layer_name=None):
        for name, param in self.named_parameters():
            if layer_name == "all" or layer_name in name:
                param.requires_grad = True

    def _inject_at_input(
        self,
        input_ids: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        past_covariates: torch.Tensor = None,
        future_covariates: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        # input injection
        inputs_embeds = None
        decoder_inputs_embeds = None
        if input_ids is not None:
            inputs_embeds = self.encoder.embed_tokens(input_ids)
            inputs_embeds = self.input_injection_block(inputs_embeds, past_covariates)
            input_ids = None

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        if future_covariates is not None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)

            # shifting covariates
            shifted_future_covariates = self._shift_right(
                future_covariates.transpose(1, 2)
            ).transpose(1, 2)

            decoder_inputs_embeds = self.input_injection_block(
                decoder_inputs_embeds, shifted_future_covariates, is_decoder=True
            )
            decoder_input_ids = None

        return (input_ids, decoder_input_ids, inputs_embeds, decoder_inputs_embeds)

    def _inject_at_output(self, output, labels, future_covariates):
        last_hidden_state = output.decoder_hidden_states[-1]
        if future_covariates is not None:
            output.logits, output.loss = self.output_injection_block(
                future_covariates=future_covariates,
                labels=labels,
                logits=output.logits,
                last_hidden_state=last_hidden_state,
            )

        return output

    def forward(
        self,
        input_ids: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        decoder_inputs_embeds: torch.Tensor = None,
        past_covariates: torch.Tensor = None,
        future_covariates: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):

        if self.input_injection_block is not None:
            (input_ids, decoder_input_ids, inputs_embeds, decoder_inputs_embeds) = (
                self._inject_at_input(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    past_covariates=past_covariates,
                    future_covariates=future_covariates,
                    labels=labels,
                )
            )

        # Removing key 'num_items_in_batch' from kwargs.
        # This is necessary as recent versions of transformers make the code break with it.
        kwargs_filtered = kwargs.copy()
        if "num_items_in_batch" in kwargs_filtered:
            kwargs_filtered.pop("num_items_in_batch")

        output = super(ChronosX, self).forward(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_hidden_states=self.output_hidden_states,
            **kwargs_filtered,
        )

        if self.output_injection_block is not None:
            output = self._inject_at_output(
                output=output, labels=labels, future_covariates=future_covariates
            )

        return output

    def generate(self, **kwargs):
        if self.output_injection_block is not None:
            self.output_injection_block.restart_generator_counter()

        if self.input_injection_block is not None:
            self.input_injection_block.restart_generator_counter()
            self.input_injection_block_decoder.restart_generator_counter()

        output = super(ChronosX, self).generate(**kwargs)

        if self.output_injection_block is not None:
            self.output_injection_block.generating = False

        if self.input_injection_block is not None:
            self.input_injection_block.generating = False
            self.input_injection_block_decoder.generating = False

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        future_covariates=None,
        **kwargs,
    ):

        kwargs_augmented = {
            **kwargs,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "encoder_outputs": encoder_outputs,
        }

        output = super(ChronosX, self).prepare_inputs_for_generation(
            input_ids,
            **kwargs_augmented,
        )

        output.update({"future_covariates": future_covariates})

        return output

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str],
        generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = (
            "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        )
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value
                for argument, value in encoder_kwargs.items()
                if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = (
            model_input_name if model_input_name is not None else self.main_input_name
        )
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore

        # IMPORTANT: here is the place where we provide the updated token embeddings for Input Injection Block
        # Remember that we inject covariates on the token embeddings. This means that
        # we do not have to give to the encoder the input_ids but rather the updated token embeddings.
        # That's why we remove input_ids from *encoder_kwargs* and rather use 'inputs_embeds'
        if self.input_injection_block is not None:
            inputs_embeds = self.encoder.embed_tokens(encoder_kwargs["input_ids"])
            inputs_embeds = self.input_injection_block(
                inputs_embeds, model_kwargs["past_covariates"]
            )
            encoder_kwargs.pop("input_ids")
            encoder_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)  # type: ignore

        return model_kwargs
