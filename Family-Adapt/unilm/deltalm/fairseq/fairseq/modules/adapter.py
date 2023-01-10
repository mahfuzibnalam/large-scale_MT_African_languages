# Original code from adapter-transformers
# https://github.com/Adapter-Hub/adapter-transformers

"""Adapter modules."""
import math
from collections import OrderedDict

import torch
from torch import nn


class Activation_Function_Class(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.nn.functional.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="relu",
        init_bert_weights=True,
        add_layer_norm_before=True,
        add_layer_norm_after=False,
        residual_before_ln=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.down_sample = down_sample
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln

        seq_list = []

        if self.add_layer_norm_before:
            seq_list.append(nn.LayerNorm(self.input_size))

        if down_sample is None:
            self.down_sample = self.input_size // 2

        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        seq_list.append(Activation_Function_Class(non_linearity.lower()))

        self.adapter_down = nn.Sequential(*seq_list)

        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if init_bert_weights:
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)


    def forward(self, x, residual_input):

        down = self.adapter_down(x)
        up = self.adapter_up(down)
        output = up

        if self.residual_before_ln:
            output = output + residual_input

        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if not self.residual_before_ln:
            output = output + residual_input

        return output, down, up

    @staticmethod
    def init_bert_weights(module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()