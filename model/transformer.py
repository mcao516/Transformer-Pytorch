#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implement Transformer modules.

   Author: Meng Cao
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    """Clone N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):
    """Implement the position embedding function.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        # position: [max_len, 1]
        position = torch.arange(0., max_len).unsqueeze(1)
        # div_term: [d_model / 2]
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EmbeddingLayer(nn.Module):
    """Inplement embedding layer.
    """
    def __init__(self, d_model, vocab_size, dropout=0.1):
        super(EmbeddingLayer, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        e = self.embed(x) * math.sqrt(self.d_model)
        e = self.position_embed(e)
        return self.dropout(e)


class Encoder(nn.Module):
    """The encoder is composed of a stack of N = 6 identical layers.
    """
    def __init__(self, d_model, N, head_num, d_ff, dropout=0.1, last_norm=True):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(EncoderLayer(MultiHeadAttentioin(d_model, head_num, dropout=dropout),
                                          LayerNorm(d_model),
                                          FeedForward(d_model, d_ff, dropout=dropout)), N)
        self.norm = LayerNorm(d_model) if last_norm else None

    def forward(self, x, mask):
        """Forward through N identical layers.

        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, seq_len]
        """
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x) if self.norm else x

        return x


class EncoderLayer(nn.Module):
    """Implement one encoder layer.
    """
    def __init__(self, attn, norm, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.norm = norm
        self.feed_forward = feed_forward

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Forward through one encoder layer: multi-head attn => add & norm
           => feed forward => add & norm.

        Args:
            x: embeddings or output of the last layer.
                [batch_size, seq_len, d_model]
            mask: [batch_size, (1 or seq_len), seq_len]
        """
        # multihead attn & norm
        a = self.attn(x, x, x, mask)
        t = self.norm(x + self.dropout(a))

        # feed forward & norm
        z = self.feed_forward(t)  # linear(dropout(act(linear(x)))))
        y = self.norm(t + self.dropout(z))

        assert x.shape == y.shape
        return y


class MultiHeadAttentioin(nn.Module):
    """Implement a multi-head attention layer.
    """
    def __init__(self, d_model, head_num, dropout=0.1):
        super(MultiHeadAttentioin, self).__init__()
        assert d_model % head_num == 0, "d_model must be divisible by head_num"

        self.d_model = d_model
        self.head_num = head_num
        self.d_k = d_model // head_num

        # d_model = d_k * head_num
        self.W_Q = nn.Linear(d_model, head_num * self.d_k)
        self.W_K = nn.Linear(d_model, head_num * self.d_k)
        self.W_V = nn.Linear(d_model, head_num * self.d_k)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dp_attn(self, query, key, value, mask=None):
        """Compute Scaled Dot-Product Attention function.

        Args:
            query: [batch_size, head_num, seq_len, d_k]
            key:   [batch_size, head_num, seq_len, d_k]
            value: [batch_size, head_num, seq_len, d_k]
            mask:  [batch_size, (1 or seq_len), seq_len]
        """
        assert self.d_k == query.shape[-1]

        # scores: [batch_size, head_num, seq_len, seq_len]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            assert mask.ndim == 3, "Mask shape doesn't seem right..."
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # attn: [batch_size, head_num, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value), attn

    def forward(self, q, k, v, mask):
        """First linearly proj the queries, keys and values, then apply
           scaled dot-product attention.

        Args:
            q: embeddings or output of the last layer.
                [batch_size, seq_len, d_model]
            mask: [batch_size, (1 or seq_len), seq_len]
        """
        batch_size = q.shape[0]

        # query, key, value: [batch_size, head_num, seq_len, d_k]
        query = self.W_Q(q).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        key = self.W_K(k).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        value = self.W_V(v).view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)

        # attn: [batch_size, head_num, seq_len, seq_len]
        heads, attn = self.scaled_dp_attn(query, key, value, mask)
        heads = heads.transpose(1, 2).contiguous().view(batch_size, -1,
                                                        self.head_num * self.d_k)
        # heads: [batch_size, seq_len, d_model]
        assert heads.shape[-1] == self.d_model and heads.shape[0] == batch_size

        # Concat(head_1, ..., head_n)W_O
        y = self.W_O(heads)

        assert y.shape == q.shape
        return y


class LayerNorm(nn.Module):
    """Construct a layernorm module.
    """
    def __init__(self, layer_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(layer_size))
        self.b = nn.Parameter(torch.zeros(layer_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.g * x + self.b


class FeedForward(nn.Module):
    """Implement a feed-forward neural network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, act='relu'):
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.ffn_1 = nn.Linear(d_model, d_ff)
        self.ffn_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        else:
            raise ValueError("Unknown activation function type.")

    def forward(self, x):
        """FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

        Args:
            x: [batch_size, seq_len, d_model]
        """
        y = self.ffn_2(self.dropout(self.act(self.ffn_1(x))))
        return y


class DecoderLayer(nn.Module):
    """Implement one decoder layer.
    """
    def __init__(self, self_attn, ed_attn, feed_forward, norm, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.ed_attn = ed_attn
        self.feed_forward = feed_forward
        self.norm = norm

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, memory, src_mask):
        """Forward through a decoder: self attn & norm =>
           encoder-decoder attn & norm => feed forward & norm.
        """
        # self-attn & norm
        a_1 = self.self_attn(x, x, x, mask)
        t_1 = self.norm(x + self.dropout(a_1))

        # encoder-decoder attn & norm
        a_2 = self.ed_attn(t_1, memory, memory, src_mask)
        t_2 = self.norm(t_1 + self.dropout(a_2))

        # feed forward & norm
        h = self.feed_forward(t_2)
        y = self.norm(t_2 + self.dropout(h))

        assert x.shape == y.shape, "Input and output shape should be the same!"
        return y


class Decoder(nn.Module):
    """Implement Transformer decoder.
    """
    def __init__(self, d_model, N, head_num, d_ff, dropout=0.1, last_norm=True):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(MultiHeadAttentioin(d_model, head_num, dropout=dropout),
                                          MultiHeadAttentioin(d_model, head_num, dropout=dropout),
                                          FeedForward(d_model, d_ff, dropout=dropout),
                                          LayerNorm(d_model)), N)
        self.norm = LayerNorm(d_model) if last_norm else None

    def forward(self, x, mask, memory, src_mask):
        """Forward through N identical decoder layers.

        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len]
            memory: [batch_size, seq_len, d_model]
            src_mask: [batch_size, 1, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask, memory, src_mask)
        x = self.norm(x) if self.norm else x

        return x


class LinearSoftmax(nn.Module):
    """Implement the final linear layer.
    """
    def __init__(self, d_model, vocab):
        super(LinearSoftmax, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, prob=True):
        """
        Args:
           x: [batch_size, seq_len, d_model]
           prob: if calculate probabilities.
        """
        logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)


def build_mask(base_mask):
    """Build a mask for the Transformer decoder to mask all the
       subsequent tokens.

    Args:
       base_mask: basic mask for padded tokens. [batch_size, seq_len]
    """
    assert len(base_mask.shape) == 2
    batch_size, seq_len = base_mask.shape[0], base_mask.shape[-1]

    # create subsequent token mask
    sub_mask = torch.tril(torch.ones([seq_len, seq_len],
                                     dtype=torch.uint8)).type_as(base_mask)
    sub_mask = sub_mask.unsqueeze(0).expand(batch_size, -1, -1)
    base_mask = base_mask.unsqueeze(1).expand(-1, seq_len, -1)
    return sub_mask & base_mask


class EncoderDecoder(nn.Module):
    """Implement an encoder-decoder architecture.
    """
    def __init__(self, src_embed, tgt_embed, encoder, decoder, linear_softmax):
        super(EncoderDecoder, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.linear_softmax = linear_softmax

    def forward(self, en_input, en_mask, de_input, de_mask):
        """Forward through the whole encoding & decoing process:
           token embedding => position embedding => encoding =>
           decoding => linear & softmax => probs

        Args:
           en_input: source tokens. [batch_size, en_seq_len]
           en_mask: source mask. [batch_size, en_seq_len]
           de_input: target tokens. [batch_size, de_seq_len]
           de_mask: target mask. [batch_size, de_seq_len]
        """
        # build masks
        en_mask = en_mask.unsqueeze(1)
        de_mask = build_mask(de_mask)

        # token & position embedding
        en_embeddings = self.src_embed(en_input)
        de_embeddings = self.tgt_embed(de_input)

        # encoding & decoding
        en_output = self.encoder(en_embeddings, en_mask)
        de_output = self.decoder(de_embeddings, de_mask, en_output, en_mask)

        # linear & softmax
        probs = self.linear_softmax(de_output)
        return probs
