import torch
import torch.nn as nn
import pickle
import pdb
from torch.autograd import Variable
from module.Hypergraph import Hypergraph
import numpy as np


class SemiMention(nn.Module):
    """
    Only dynamic Embedding, Self RNN, Hypergraph
    """
    def __init__(self, config):
        super(SemiMention, self).__init__()
        self.config = config

        if self.config.if_pos:
            self.pos_embed = nn.Embedding(config.pos_size, config.pos_embed)
            self.rnn = nn.LSTM(config.token_embed + config.pos_embed, config.f_hidden_size, batch_first = True, 
               num_layers = config.f_layers, dropout = config.f_lstm_dropout, bidirectional = True)
        else:
            self.rnn = nn.LSTM(config.token_embed, config.f_hidden_size, batch_first = True, 
               num_layers = config.f_layers, dropout = config.f_lstm_dropout, bidirectional = True)

        self.word_embed = nn.Embedding(config.voc_size, config.token_embed)
        self.input_dropout = nn.Dropout(config.input_dropout)

        self.hypergraph = Hypergraph(config)

    def forward(self, token_batch, pos_batch, label_batch):
        word_vec = self.word_embed(token_batch)
        if self.config.if_pos:
            pos_vec = self.pos_embed(pos_batch)
            word_cat = torch.cat([word_vec, pos_vec], 2)
        else:
            word_cat = word_vec
        word_cat = self.input_dropout(word_cat)

        feat2hyper = None
        if self.config.if_interactions:
            lstm_out, (hid_states, cell_states) = self.rnn(word_cat)
            feat2hyper = lstm_out
        else:
            feat2hyper = word_cat

        ret_dic = self.hypergraph(feat2hyper, label_batch)
        return ret_dic["loss"]

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            t_v = torch.Tensor(vectors)
            print("Loading from {} with size {}".format(self.config.embed_path, t_v.size()))
            self.word_embed.weight = nn.Parameter(t_v)
            # self.word_embed.weight.requires_grad = False

    def predict(self, token_batch, pos_batch):
        word_vec = self.word_embed(token_batch)
        if self.config.if_pos:
            pos_vec = self.pos_embed(pos_batch)
            word_cat = torch.cat([word_vec, pos_vec], 2)
        else:
            word_cat = word_vec
        word_cat = self.input_dropout(word_cat)

        feat2hyper = None
        if self.config.if_interactions:
            lstm_out, (hid_states, cell_states) = self.rnn(word_cat)
            feat2hyper = lstm_out
        else:
            feat2hyper = word_cat

        return self.hypergraph.decode(feat2hyper)
