import random
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig
import pdb
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)

class OPTForCausalLM(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        #self.model = OPTModel(config)
        #self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.post_init()

        # Embedding parameters
        self.model_decoder_embed_tokens_weight = nn.Parameter(torch.Tensor(config.vocab_size, config.word_embed_proj_dim))
        self.model_decoder_embed_positions_weight = nn.Parameter(torch.Tensor(config.max_position_embeddings+2, config.word_embed_proj_dim))

        # Final layer normalization parameters
        self.model_decoder_final_layer_norm_weight = nn.Parameter(torch.Tensor(config.word_embed_proj_dim))
        self.model_decoder_final_layer_norm_bias = nn.Parameter(torch.Tensor(config.word_embed_proj_dim))

        # Initialize parameters for each of the 12 decoder layers
        for i in range(config.num_hidden_layers):
            setattr(self, f"model_decoder_layers_{i}_self_attn_k_proj_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_k_proj_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_v_proj_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_v_proj_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_q_proj_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_q_proj_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_out_proj_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_out_proj_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_layer_norm_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_self_attn_layer_norm_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_fc1_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.ffn_dim)))
            setattr(self, f"model_decoder_layers_{i}_fc1_bias", nn.Parameter(torch.Tensor(config.ffn_dim)))
            setattr(self, f"model_decoder_layers_{i}_fc2_weight", nn.Parameter(torch.Tensor(config.ffn_dim, config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_fc2_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_final_layer_norm_weight", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
            setattr(self, f"model_decoder_layers_{i}_final_layer_norm_bias", nn.Parameter(torch.Tensor(config.word_embed_proj_dim)))
        self.lm_head_weight = nn.Parameter(torch.Tensor(config.word_embed_proj_dim, config.vocab_size))
        #self.model_decoder_layers_0_self_attn_k_proj_weight
        # Initialize all weights
        #self.apply(self.init_weights)

    #def forward(self, input_ids, attention_mask=None):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #inputs_embeds = self.embed_tokens_weight[input_ids] + self.embed_positions_weight[:input_ids.size(1), :]

        #self.lm_head_weight.data = self.model_decoder_embed_tokens_weight.data.contiguous()

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions #F opt125M
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states #F opt125M
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict #T opt125M

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self._decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ) #[0] [32,24,768] [1] .. opt125M
        #logits = self.lm_head(outputs[0]).contiguous()  # [32,24,50272]  opt125M
        logits = nn.functional.linear(outputs[0], self.lm_head_weight).contiguous()
        loss = None
        if labels is not None: #F opt125M
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict: #F opt125M
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(loss=loss,logits=logits,past_key_values=outputs.past_key_values,hidden_states=outputs.hidden_states,attentions=outputs.attentions,)

    def _decoder(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
            combined_attention_mask = None
            if input_shape[-1] > 1:
                combined_attention_mask = _make_causal_mask(
                    input_shape,
                    inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    past_key_values_length=past_key_values_length,
                )
            if attention_mask is not None:
                expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(inputs_embeds.device)
                combined_attention_mask = (
                    expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask)
            return combined_attention_mask

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions#F
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states#F
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache#T

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict#T

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:#T
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:#T
            #inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = nn.functional.embedding(input_ids, self.model_decoder_embed_tokens_weight, self.config.pad_token_id)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:#F
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        causal_attention_mask = _prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        #pos_embeds = self.embed_positions(attention_mask, past_key_values_length)  #OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        attention_mask = attention_mask.long()
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        positions = positions[:, past_key_values_length:]
        pos_embeds = nn.functional.embedding(positions+2, self.model_decoder_embed_positions_weight)

        #if self.project_in is not None:#F
        #    inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        

        self.gradient_checkpointing = False
        self.layerdrop = self.config.layerdrop

        if self.gradient_checkpointing and self.training:#F
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        #for idx, decoder_layer in enumerate(self.layers):
        for layer_index in range(self.config.num_hidden_layers):
            if output_hidden_states:#F
                all_hidden_states += (hidden_states,)

            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:#F

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self._decoder_layer),
                    hidden_states,
                    causal_attention_mask,
                    head_mask[layer_index] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = self._decoder_layer(
                    layer_index,
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[layer_index] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            
            if use_cache:#T
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:#F
                all_self_attns += (layer_outputs[1],)
        #pdb.set_trace()
        #if self.config.final_layer_norm is not None:#T
        if self.config.do_layer_norm_before and not self.config._remove_final_layer_norm:
            #hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = nn.functional.layer_norm(hidden_states.to(self.model_decoder_final_layer_norm_weight.device), hidden_states.shape[-1:], self.model_decoder_final_layer_norm_weight, self.model_decoder_final_layer_norm_bias)
        #if self.project_out is not None:#F
        #    hidden_states = self.project_out(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:#F
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache,hidden_states=all_hidden_states,attentions=all_self_attns,)

    def _decoder_layer(self, layer_index,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.config.do_layer_norm_before:#T config.do_layer_norm_before
            #hidden_states = self.self_attn_layer_norm(hidden_states)
            self_attn_layer_norm_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_layer_norm_weight")
            self_attn_layer_norm_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_layer_norm_bias")
            #pdb.set_trace()
            hidden_states = nn.functional.layer_norm(hidden_states.to(self_attn_layer_norm_weight.device), hidden_states.shape[-1:], self_attn_layer_norm_weight, self_attn_layer_norm_bias)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self._self_attn(layer_index, 
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)
        hidden_states = residual.to(hidden_states.device) + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.config.do_layer_norm_before:
            #hidden_states = self.self_attn_layer_norm(hidden_states)
            self_attn_layer_norm_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_layer_norm_weight")
            self_attn_layer_norm_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_layer_norm_bias")
            hidden_states = nn.functional.layer_norm(hidden_states, hidden_states.shape[-1:], self_attn_layer_norm_weight, self_attn_layer_norm_bias)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.config.do_layer_norm_before:
            #hidden_states = self.final_layer_norm(hidden_states)
            final_layer_norm_weight = getattr(self, f"model_decoder_layers_{layer_index}_final_layer_norm_weight")
            final_layer_norm_bias = getattr(self, f"model_decoder_layers_{layer_index}_final_layer_norm_bias")
            hidden_states = nn.functional.layer_norm(hidden_states, hidden_states.shape[-1:], final_layer_norm_weight, final_layer_norm_bias)

        self.activation_fn = ACT2FN[self.config.activation_function]

        #hidden_states = self.fc1(hidden_states)
        fc1_weight = getattr(self, f"model_decoder_layers_{layer_index}_fc1_weight")
        fc1_bias = getattr(self, f"model_decoder_layers_{layer_index}_fc1_bias")
        hidden_states = nn.functional.linear(hidden_states, fc1_weight, fc1_bias)
        hidden_states = self.activation_fn(hidden_states)

        fc2_weight = getattr(self, f"model_decoder_layers_{layer_index}_fc2_weight")
        fc2_bias = getattr(self, f"model_decoder_layers_{layer_index}_fc2_bias")
        hidden_states = nn.functional.linear(hidden_states, fc2_weight, fc2_bias)
        #hidden_states = self.activation_fn(hidden_states)

        #hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.config.do_layer_norm_before:
            #hidden_states = self.final_layer_norm(hidden_states)
            final_layer_norm_weight = getattr(self, f"model_decoder_layers_{layer_index}_final_layer_norm_weight")
            final_layer_norm_bias = getattr(self, f"model_decoder_layers_{layer_index}_final_layer_norm_bias")
            hidden_states = nn.functional.layer_norm(hidden_states, hidden_states.shape[-1:], final_layer_norm_weight, final_layer_norm_bias)
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


    def _self_attn(self, layer_index,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        self.num_heads = self.config.num_attention_heads
        self.head_dim = self.config.hidden_size // self.num_heads  

        def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
            return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        #query_states = self.q_proj(hidden_states) * self.scaling
        q_proj_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_q_proj_weight")
        q_proj_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_q_proj_bias")
        query_states = nn.functional.linear(hidden_states, q_proj_weight, q_proj_bias)
        query_states = query_states * self.head_dim**-0.5

        k_proj_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_k_proj_weight")
        k_proj_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_k_proj_bias")
        

        v_proj_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_v_proj_weight")
        v_proj_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_v_proj_bias")
        

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            #key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            #value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            key_states = nn.functional.linear(key_value_states, k_proj_weight, k_proj_bias)
            value_states = nn.functional.linear(key_value_states, v_proj_weight, v_proj_bias)
            key_states = _shape(key_states, -1, bsz)
            value_states = _shape(value_states, -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            #key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            #value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = nn.functional.linear(hidden_states, k_proj_weight, k_proj_bias)
            value_states = nn.functional.linear(hidden_states, v_proj_weight, v_proj_bias)
            key_states = _shape(key_states, -1, bsz)
            value_states = _shape(value_states, -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            #key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            #value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = nn.functional.linear(hidden_states, k_proj_weight, k_proj_bias)
            value_states = nn.functional.linear(hidden_states, v_proj_weight, v_proj_bias)
            key_states = _shape(key_states, -1, bsz)
            value_states = _shape(value_states, -1, bsz)

        self.is_decoder = True
        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = _shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask.to(attn_weights.device)
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.config.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.config.hidden_size)

        #attn_output = self.out_proj(attn_output)
        out_proj_weight = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_out_proj_weight")
        out_proj_bias = getattr(self, f"model_decoder_layers_{layer_index}_self_attn_out_proj_bias")
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        return attn_output, attn_weights_reshaped, past_key_value

    

'''
model.decoder.embed_tokens.weight', 'model.decoder.embed_positions.weight', 'model.decoder.final_layer_norm.weight',
 'model.decoder.final_layer_norm.bias', 'model.decoder.layers.0.self_attn.k_proj.weight', 'model.decoder.layers.0.self_attn.k_proj.bias',
  'model.decoder.layers.0.self_attn.v_proj.weight', 'model.decoder.layers.0.self_attn.v_proj.bias', 'model.decoder.layers.0.self_attn.q_proj.weight',
   'model.decoder.layers.0.self_attn.q_proj.bias', 'model.decoder.layers.0.self_attn.out_proj.weight', 'model.decoder.layers.0.self_attn.out_proj.bias',
    'model.decoder.layers.0.self_attn_layer_norm.weight', 'model.decoder.layers.0.self_attn_layer_norm.bias', 'model.decoder.layers.0.fc1.weight', 
    'model.decoder.layers.0.fc1.bias', 'model.decoder.layers.0.fc2.weight', 'model.decoder.layers.0.fc2.bias', 'model.decoder.layers.0.final_layer_norm.weight',
     'model.decoder.layers.0.final_layer_norm.bias', 
'''

'''
OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 768, padding_idx=1)
      (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
      (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (layers): ModuleList(
        (0-11): 12 x OPTDecoderLayer(
          (self_attn): OPTAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=768, out_features=3072, bias=True)
          (fc2): Linear(in_features=3072, out_features=768, bias=True)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)
  (lm_head): Linear(in_features=768, out_features=50272, bias=False)
)
'''
