from torch import nn
from torch.autograd import Variable
import torch
import math
from onmt.Models import DecoderState

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(EncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma = MultiheadAttention(hidden_size,
                                               hidden_size,
                                               hidden_size,
                                               dropout)
        self.ffn = ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_prenorm = LayerNorm(hidden_size)
        self.ffn_prenorm = LayerNorm(hidden_size)
        self.ma_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, bias):
        """
        Args:
            x: batch * length * channels
            bias: batch * length * length
        """
        # multihead attention
        y, _ = self.ma(self.ma_prenorm(x), None, self.num_heads, bias)
        x = self.ma_postdropout(y) + x
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size,
                 dropout, embeddings, uni_vecs):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.layer_stack = nn.ModuleList([
            EncoderLayer(hidden_size, dropout) for _ in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.vecs = uni_vecs ## A universal space to be accessed by embedding.
        if self.vecs is not None:
            self.num_heads = 8
            self.ma_prenorm = LayerNorm(hidden_size)
            self.ma = MultiheadAttention(hidden_size,
                                         hidden_size,
                                         hidden_size,
                                         dropout,
                                         trans=False)
            self.ma_postdropout = nn.Dropout(dropout)
            self.softmax = nn.Softmax(dim=-1)
            self.attn_dp = nn.Dropout(dropout)
            if not self.vecs.requires_grad:
                dim = self.vecs.shape[1]
                self.w = nn.Linear(dim, dim)


    def forward(self, input, src_lengths):
        ### Seems like size of input must end with 1

        emb = self.embeddings(input)
        # s_len, n_batch, emb_dim = emb.size()
        out = emb.transpose(0, 1).contiguous()

        if self.vecs is not None:
            mid = self.ma_prenorm(out)
            if not self.vecs.requires_grad:
                v = self.w(self.vecs)
            else:
                v = self.vecs
            v = v.unsqueeze(0).expand(out.size(0), -1, -1).contiguous()
            # mid, _ = self.ma(mid, v, self.num_heads, None)
            att = torch.matmul(mid, v.transpose(1, 2).contiguous())
            att = self.softmax(att)
            drop_att = self.attn_dp(att)
            mid = torch.matmul(drop_att, v)
            out = self.ma_postdropout(mid) + out
        if input.dim() == 4:
            words = input[:, :, 0, 0]
        else:
            words = input[:, :, 0].transpose(0, 1)


        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).float()
        bias = Variable(torch.unsqueeze(mask * -1e9, 1))
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.layer_stack[i](out, bias)
        out = self.layer_norm(out)

        # enc_final & memory_bank
        return Variable(emb.data), out.transpose(0, 1).contiguous()


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout,
                 num_heads=8, filter_size=2048):
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.ma_l1 = MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ma_l2 = MultiheadAttention(hidden_size,
                                                  hidden_size,
                                                  hidden_size,
                                                  dropout)
        self.ffn = ffn_layer(hidden_size,
                                    filter_size,
                                    hidden_size,
                                    dropout)
        self.ma_l1_prenorm = LayerNorm(hidden_size)
        self.ma_l2_prenorm = LayerNorm(hidden_size)
        self.ffn_prenorm = LayerNorm(hidden_size)
        self.ma_l1_postdropout = nn.Dropout(dropout)
        self.ma_l2_postdropout = nn.Dropout(dropout)
        self.ffn_postdropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, self_attention_bias,
                encoder_decoder_bias, previous_input=None):
        # self multihead attention
        norm_x = self.ma_l1_prenorm(x)
        all_inputs = norm_x
        if previous_input is not None:
            all_inputs = torch.cat((previous_input, norm_x), dim=1)
            self_attention_bias = None
        y, _ = self.ma_l1(norm_x, all_inputs, self.num_heads, self_attention_bias)
        x = self.ma_l1_postdropout(y) + x
        # encoder decoder multihead attention
        y, attn = self.ma_l2(self.ma_l2_prenorm(x), encoder_output,
                             self.num_heads, encoder_decoder_bias)
        x = self.ma_l2_postdropout(y) + x
        # ffn layer
        y = self.ffn(self.ffn_prenorm(x))
        ans = self.ffn_postdropout(y) + x
        return ans, attn, all_inputs


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, hidden_size, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        # Basic attributes.
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.layer_stack = nn.ModuleList([
                DecoderLayer(hidden_size, dropout) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()

        src = state.src
        if src.dim() == 4:
            src_words = src[:, :, 0, 0]
        else:
            src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()

        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)

        # Initialize return variables.
        attns = {"std": []}

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):, ]
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()  # batch * length * channels
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = Variable(src_words.data.eq(padding_idx).float())
        tgt_pad_mask = Variable(tgt_words.data.eq(padding_idx).float().unsqueeze(1))
        tgt_pad_mask = tgt_pad_mask.repeat(1, tgt_len, 1)  # batch * tgt_len * tgt_len
        encoder_decoder_bias = torch.unsqueeze(src_pad_mask * -1e9, 1)  # batch * 1 * src_length
        decoder_local_mask = get_local_mask(tgt_len)  # [1, length, length]
        decoder_local_mask = decoder_local_mask.repeat(tgt_batch, 1, 1)
        decoder_bias = torch.gt(tgt_pad_mask + decoder_local_mask, 0).float() * -1e9  # batch * tgt_len * tgt_len

        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]

            output, attn, all_input \
                    = self.layer_stack[i](output, src_memory_bank, decoder_bias,
                                          encoder_decoder_bias, previous_input=prev_layer_input)
            saved_inputs.append(all_input)

        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        # attn batch_size * tgt_lengths * src_lengths
        # batch * src_length * tgt_length
        attn = attn.transpose(0, 1).contiguous()

        attns["std"] = attn

        # Update the state.
        state = state.update_state(tgt, saved_inputs)

        return outputs, state, attns

    def init_decoder_state(self, src):
        return TransformerDecoderState(src)

class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None

    @property
    def _all(self):
        return (self.previous_input, self.previous_layer_inputs, self.src)

    def update_state(self, input, previous_layer_inputs):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src)
        state.previous_input = input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1), volatile=True)

class MultiheadAttention(nn.Module):
    def __init__(self,
                 total_key_depth,
                 total_value_depth,
                 channels,
                 attention_dropout=0.0,
                 log_softmax=False,
                 trans=True):
        super(MultiheadAttention, self).__init__()
        self.total_key_depth = total_key_depth
        self.trans = trans
        if trans:
            self.input_query_transform = nn.Linear(channels, total_key_depth)
            self.input_key_transform = nn.Linear(channels, total_key_depth)
            self.input_value_transform = nn.Linear(channels, total_value_depth)
        self.attention_softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_transform = nn.Linear(total_value_depth, channels)

        self.use_log_softmax = log_softmax
        if log_softmax:
            self.log_softmax = nn.LogSoftmax(dim=-1)

    def split_heads(self, x, num_heads):
        """
        Args:
            x: a Tensor with shape [batch, length, channels]
            num_heads: a int number, channels of x should be devided by num_heads
        Returns:
            ans: a Tensor with shape [batch * num_heads, length, channels // num_heads]
        """
        batch, length, channels = x.size()
        assert channels % num_heads == 0, (
               "channels of the input should be devided by num_heads")
        new_dim = channels // num_heads
        ans = x.view(batch, length, num_heads, new_dim).transpose(1, 2)
        return ans

    def combine_heads(self, x, num_heads):
        """
        A reverse process of split_heads function
        Args:
            x: batch * num_heads, length, last_dim
        Returns:
            ans: batch, length, last_dim * num_heads
        """
        batch, _, length, new_dim = x.size()
        ans = x.transpose(1, 2).contiguous().view(batch,
                                    length, num_heads * new_dim)
        return ans

    def forward(self,
                query_antecedent,
                memory_antecedent,
                num_heads,
                bias,
                weight_matrix=None):
        """
        Args:
            query_antecedent: a Tensor with shape [batch, length_q, channels]
            memory_antecedent: a Tensor with shape [batch, length_kv, channels]
            bias: bias Tensor with shape [batch, 1, length_kv] or [batch, length_q, length_kv]
            num_heads: a int number
        Returns:
            the result of the attention transformation, shape is [batch, length_q, channels]
        """
        if memory_antecedent is None:
            memory_antecedent = query_antecedent

        batch_size, query_len, _ = query_antecedent.size()
        _, key_len, _ = memory_antecedent.size()
        if self.trans:
            q = self.input_query_transform(query_antecedent)
            k = self.input_key_transform(memory_antecedent)
            v = self.input_value_transform(memory_antecedent)
        else:
            q = query_antecedent
            k = memory_antecedent
            v = memory_antecedent
        q = self.split_heads(q, num_heads)
        k = self.split_heads(k, num_heads)
        v = self.split_heads(v, num_heads)
        key_depth_per_head = self.total_key_depth // num_heads
        q = q / math.sqrt(key_depth_per_head)
        logits = torch.matmul(q, k.transpose(2, 3))
        # tgt_self_attention: batch * num_heads * tgt_length * tgt_length
        if weight_matrix is not None:
            weight_matrix = weight_matrix.unsqueeze(1).expand_as(logits)
            logits = logits * weight_matrix
        if bias is not None:
            bias = bias.unsqueeze(1).expand_as(logits)
            logits += bias

        attn = self.attention_softmax(logits)
        drop_attn = self.attention_dropout(attn)
        x = torch.matmul(drop_attn, v)
        top_attn = attn.view(batch_size, num_heads, query_len, key_len)[:, 0, :, :].contiguous()
        x = self.combine_heads(x, num_heads)

        if self.use_log_softmax:
            attn = self.log_softmax(logits)
            top_attn = attn.view(batch_size, num_heads, query_len, key_len)[:, 0, :, :].contiguous()
        return self.output_transform(x), top_attn

class ffn_layer(nn.Module):
    def __init__(self,
                 input_size,
                 filter_size,
                 output_size,
                 relu_dropout=0.0):
        super(ffn_layer, self).__init__()
        self.mid_layer = nn.Linear(input_size, filter_size)
        self.out_layer = nn.Linear(filter_size, output_size)
        self.relu = nn.ReLU()
        self.relu_dropout = nn.Dropout(relu_dropout)

    def forward(self, x):
        t = self.relu(self.mid_layer(x))
        o = self.out_layer(self.relu_dropout(t))
        return o

class LayerNorm(nn.Module):
    def __init__(self, depth, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(depth), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(depth), requires_grad=True)

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        norm_x = (x - mean) / (std + self.eps)
        return norm_x * self.scale + self.bias

def get_local_mask(length, diagonal=1, cuda=True):
    ans = Variable(torch.ones(length, length))
    ans = torch.triu(ans, diagonal).unsqueeze(0)
    if cuda:
        ans = ans.cuda()
    return ans
