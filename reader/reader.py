import numpy as np
import pickle
import pdb
import torch
import string
from tqdm import tqdm
from collections import namedtuple, defaultdict, OrderedDict

FILE_PATH = {"sample":"./data/genia_sample/"}
GLOVE_FILE = "./embeddings/glove.6B.100d.txt"

SentInst = namedtuple("SentInst", "tokens chars pos entities")

class Reader():
    def __init__(self, config):
        self.config = config
        self.UNK = "#UNK#"

    def read_file(self, filename, mode="train"):
        sent_list = []
        max_len = 0
        num_thresh = 0
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line == "": # last few blank lines
                    break

                raw_tokens = line.split()
                chars = [list(t) for t in raw_tokens]
                # TODO: chars is cap sensitive, token is not
                tokens = raw_tokens
                # tokens = [t.lower() for t in raw_tokens]
                # tokens = line.split()
                pos = next(f).strip().split()
                pos = [p.split('|')[0] for p in pos]

                if self.config.if_backward:
                    tokens = list(reversed(tokens))
                    pos = list(reversed(pos))
                    chars = list(reversed(chars))

                assert len(tokens) == len(pos)
                entities = next(f).strip()
                if entities == "": # no entities
                    sentInst = SentInst(tokens, chars, pos, [])
                else:
                    entity_list = []
                    entites = entities.split("|")
                    for item in entites:
                        pointers, label = item.split()
                        pointers = pointers.split(",")
                        if int(pointers[1]) > len(tokens):
                            pdb.set_trace()
                        # end - 1 inclusive
                        span_len = int(pointers[1]) - int(pointers[0])
                        if span_len > max_len:
                            max_len = span_len
                        if span_len > 6:
                            num_thresh += 1

                        if self.config.if_backward:
                            sent_len = len(tokens)
                            new_entity = ( sent_len - int(pointers[1]), sent_len - int(pointers[0]) - 1, label)
                        else:
                            new_entity = (int(pointers[0]), int(pointers[1]) - 1, label)
                        # may be dumplicate entities in some datasets
                        if (mode == "train" and new_entity not in entity_list) or (mode != "train"):
                            entity_list.append(new_entity)

                    # assert len(entity_list) == len(set(entity_list)) # check duplicate
                    sentInst = SentInst(tokens, chars, pos, entity_list)
                assert next(f).strip() == "" # seperating line

                sent_list.append(sentInst)
        print("Max length: ", max_len)
        print("Threshold 6: ", num_thresh)
        return sent_list

    def gen_dic(self):
        word_set = set()
        pos_set = set()
        label_set = set()
        char_set = set()

        for sent_list in [self.train, self.dev, self.test]:
            num_mention = 0
            for sentInst in sent_list:
                for token in sentInst.chars:
                    for char in token:
                        char_set.add(char)
                for token in sentInst.tokens:
                    word_set.add(token)
                for pos in sentInst.pos:
                    pos_set.add(pos)
                for entity in sentInst.entities:
                    label_set.add(entity[2])
                num_mention += len(sentInst.entities)
            print("# mentions :{}".format(num_mention))

        self.id2char = list(char_set)
        self.char2id = { k:i for i,k in enumerate(self.id2char)}
        self.id2word = list(word_set)
        self.word2id = { k:i for i,k in enumerate(self.id2word)}
        self.id2pos = list(pos_set)
        self.pos2id = {k:i for i,k in enumerate(self.id2pos)}
        self.id2label = list(label_set)
        self.label2id = {k:i for i,k in enumerate(self.id2label)}

    def to_batch(self):
        """
        TODO: dev and test doesn't need to meet the batch_size
        """
        ret_list = []

        for sent_list in [self.train, self.dev, self.test]:
            token_dic = defaultdict(list)
            pos_dic = defaultdict(list)
            label_dic = defaultdict(list)
            char_dic = defaultdict(list)
            char_len_dic = defaultdict(list)

            this_token_batches = []
            this_pos_batches = []
            this_label_batches = []
            this_char_batches = []
            this_char_len_batches = []

            for sentInst in sent_list:
                char_mat = [ [self.char2id[c] for c in t] for t in sentInst.chars ]
                # max_len = max([len(t) for t in sentInst.chars])
                # char_mat = [ t + [0] * (max_len - len(t)) for t in char_mat ]

                char_len_vec = [ len(t) for t in sentInst.chars ]
                token_vec = [self.word2id[t] for t in sentInst.tokens]
                pos_vec = [self.pos2id[p] for p in sentInst.pos]
                label_list = [ (u[0], u[1], self.label2id[u[2]]) for u in sentInst.entities ]
                token_dic[len(sentInst.tokens)].append(token_vec)
                pos_dic[len(sentInst.tokens)].append(pos_vec)
                label_dic[len(sentInst.tokens)].append(label_list)
                char_dic[len(sentInst.tokens)].append(char_mat)
                char_len_dic[len(sentInst.tokens)].append(char_len_vec)

            for length in token_dic.keys():
                token_batches = [token_dic[length][i : i + self.config.batch_size] for i in range(0, len(token_dic[length]), self.config.batch_size)]
                pos_batches = [pos_dic[length][i : i + self.config.batch_size] for i in range(0, len(pos_dic[length]), self.config.batch_size)]
                label_batches = [label_dic[length][i : i + self.config.batch_size] for i in range(0, len(label_dic[length]), self.config.batch_size)]
                char_batches = [char_dic[length][i : i + self.config.batch_size] for i in range(0, len(label_dic[length]), self.config.batch_size)]
                char_len_batches = [char_len_dic[length][i : i + self.config.batch_size] for i in range(0, len(label_dic[length]), self.config.batch_size)]

                this_token_batches += token_batches
                this_pos_batches += pos_batches
                this_label_batches += label_batches
                this_char_batches += char_batches
                this_char_len_batches += char_len_batches

            ret_list.append((this_token_batches, this_char_batches, this_char_len_batches, this_pos_batches, this_label_batches))

        return tuple(ret_list)

    def read_all_data(self):
        file_path = FILE_PATH[self.config.data_set]
        self.train = self.read_file(file_path + "train.data")
        self.dev = self.read_file(file_path + "dev.data", mode="dev")
        self.test = self.read_file(file_path + "test.data", mode="test")
        self.gen_dic()
        # pdb.set_trace()

    def gen_vectors_glove(self):
        vocab_dic = {}
        with open(GLOVE_FILE) as f:
            for line in f:
                s_s = line.split()
                if s_s[0] in self.word2id:
                    vocab_dic[s_s[0]] = np.array([float(x) for x in s_s[1:]])

        unknowns = np.random.uniform(-0.01, 0.01, self.config.token_embed).astype("float32")
        ret_mat = []
        unk_counter = 0
        for token in self.id2word:
            token = token.lower()
            if token in vocab_dic:
                ret_mat.append(vocab_dic[token])
            else:
                ret_mat.append(unknowns)
                # print "Unknown token:", token
                unk_counter += 1
        ret_mat = np.array(ret_mat)
        with open(self.config.embed_path, "wb") as f:
            pickle.dump(ret_mat, f)
        print("{0} unk out of {1} vocab".format(unk_counter, len(self.id2word)))


    def debug_single_sample(self, token_v, char_mat, char_len_vec, pos_v, label_list):
        print(" ".join([ self.id2word[t] for t in token_v ]))
        print(" ".join([ self.id2pos[t] for t in pos_v]))
        for t in char_mat:
            print(" ".join([ self.id2char[c] for c in t ]))
        print(char_len_vec)
        for label in label_list:
            print(label[0], label[1], self.id2label[label[2]])

