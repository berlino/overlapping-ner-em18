from __future__ import division


class Config():
    def __init__(self):
        self.root_path = "."

        # for NN only
        self.if_margin = True
        self.beta = 3

        # for data loader
        self.data_set = "sample"
        self.lowercase = True
        self.batch_size = 32
        self.if_shuffle = True
        self.if_backward = False
        self.if_interactions = True

        # override when loading data
        self.voc_size = None
        self.pos_size = None
        self.label_size = None

        # for h
        self.token_feat_size = None # for h model
        self.span_feat_size = None # for h model
        self.t_null_id = None
        self.s_null_id = None
        self.h_hidden_size = 128

        # embed size
        self.token_embed = 100
        self.if_pos = True
        self.pos_embed = 32
        self.input_dropout = 0.5

        # for lstm
        self.f_hidden_size = 128  # 32, 64, 128, 256
        self.f_layers = 1
        self.f_lstm_dropout = 0.1 # [0,0.5]
        self.semi_hidden_size = self.f_hidden_size

        # for training
        self.embed_path = self.root_path + "/data/word_vec_{0}_{1}.pkl".format(self.data_set, self.token_embed)
        self.epoch = 500
        self.if_gpu = False
        self.opt = "Adam"
        self.lr = 0.005 # [0.3, 0.00006]
        self.l2 = 1e-4
        self.check_every = 1
        self.clip_norm = 3

        # for early stop
        self.lr_patience = 3
        self.decay_patience = 2

        self.pre_trained = True
        self.data_path = self.root_path + "/data/{0}".format(self.data_set)
        self.model_path = self.root_path + "/dumps/{0}_model.pt".format(self.data_set)

        # max length
        self.if_C = True
        self.C = 6


    def __repr__(self):
        return str(vars(self))


config = Config()
