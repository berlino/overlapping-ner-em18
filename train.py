#!/usr/bin/env python
import numpy as np
import pickle
import random
from random import shuffle
from training.util import adjust_learning_rate, clip_model_grad, create_opt, load_dynamic_config
from util.evaluate import evaluate, count_overlap, evaluate_detail
from model.SemiMention import SemiMention
from config import config
from torch.autograd import Variable
import torch
import copy
import time
import pdb

# load data
f = open(config.data_path + "_train.pkl", 'rb')
train_token_batches, train_char_batch, train_char_len_batch, train_pos_batches, train_label_batches = pickle.load(f)
f.close()
f = open(config.data_path + "_dev.pkl", 'rb')
dev_token_batches, dev_char_batch, dev_char_len_batch, dev_pos_batches, dev_label_batches = pickle.load(f)
f.close()
f = open(config.data_path + "_test.pkl", 'rb')
test_token_batches, test_char_batch, test_char_len_batch, test_pos_batches, test_label_batches = pickle.load(f)
f.close()

# misc info  
# TODO: get it better
misc_config = pickle.load(open(config.data_path + "_config.pkl", 'rb'))
load_dynamic_config(misc_config, config)
id2label = misc_config["id2label"]

ner_model = SemiMention(config)
if config.pre_trained:
    ner_model.load_vector()
if config.if_gpu and torch.cuda.is_available(): ner_model = ner_model.cuda()

parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer = create_opt(parameters, config)

print("{0} batches expected for training".format(len(train_token_batches)))
best_model = None
best_per = 0
train_all_batches = list(zip(train_token_batches, train_char_batch, train_char_len_batch, train_pos_batches, train_label_batches))
if config.if_shuffle:
    shuffle(train_all_batches)


def get_f1(model, mode):
    pred_all, pred, recall_all, recall = 0, 0, 0, 0
    f_pred_all, f_pred, f_recall_all, f_recall = 0, 0, 0, 0
    gold_cross_num = 0
    pred_cross_num = 0
    if mode == "dev":
        batch_zip = zip(dev_token_batches, dev_char_batch, dev_char_len_batch, dev_pos_batches, dev_label_batches)
    elif mode == "test":
        batch_zip = zip(test_token_batches, test_char_batch, test_char_len_batch, test_pos_batches, test_label_batches)
    else:
        raise ValueError

    for token_batch, char_batch, char_len_batch, pos_batch, label_batch in batch_zip:
        token_batch_var = Variable(torch.LongTensor(np.array(token_batch)))
        pos_batch_var = Variable(torch.LongTensor(np.array(pos_batch)))
        if config.if_gpu:
            token_batch_var = token_batch_var.cuda()
            pos_batch_var = pos_batch_var.cuda()

        model.eval()
        pred_entities = model.predict(token_batch_var, pos_batch_var)
        p_a, p, r_a, r = evaluate(label_batch, pred_entities)

        #gold_cross_num += sum(count_overlap(label_batch))
        #pred_cross_num += sum(count_overlap(pred_entities))
        gold_cross_num += 0
        pred_cross_num += 0

        pred_all += p_a
        pred += p
        recall_all += r_a
        recall += r


    print(pred_all, pred, recall_all, recall)
    f1 = 2 / ((pred_all / pred) + (recall_all / recall)) 
    print( "Precision {0}, Recall {1}, F1 {2}".format(pred / pred_all, recall / recall_all, f1) )
    # print("Prediction Crossing: ", pred_cross_num)
    # print("Gold Crossing: ", gold_cross_num)

    return f1

# Test
# f1 = get_f1(ner_model, "dev")

train_start_time = time.time()
early_counter = 0
decay_counter = 0
for e_ in range(config.epoch):
    print("Epoch: ", e_ + 1)
    batch_counter = 0
    for token_batch, char_batch, char_len_batch, pos_batch, label_batch in train_all_batches:
        batch_len = len(token_batch)
        sent_len = len(token_batch[0])

        token_batch_var = Variable(torch.LongTensor(np.array(token_batch)))
        pos_batch_var = Variable(torch.LongTensor(np.array(pos_batch)))
        if config.if_gpu:
            token_batch_var = token_batch_var.cuda()
            pos_batch_var = pos_batch_var.cuda()

        ner_model.train()
        optimizer.zero_grad()
        loss = ner_model.forward(token_batch_var, pos_batch_var, label_batch)
        loss.backward()
        clip_model_grad(ner_model, config.clip_norm)
        print("batch {0} with {1} instance and sentece length {2} loss {3}".format(
            batch_counter, batch_len, sent_len, loss.cpu().data.numpy()[0]))
        batch_counter += 1

        optimizer.step()

    if (e_+1) % config.check_every != 0:
        continue

    # evaluating dev and always save the best
    cur_time = time.time()
    f1 = get_f1(ner_model, "dev")
    print("Dev step took {} seconds".format(time.time() - cur_time))

    # early stop
    if f1 > best_per:
        early_counter = 0
        best_per = f1
        del best_model
        best_model = copy.deepcopy(ner_model)
    else:
        early_counter += 1
        if early_counter > config.lr_patience:
            decay_counter += 1
            early_counter = 0
            if decay_counter > config.decay_patience:
                break
            else:
                adjust_learning_rate(optimizer)
print("")
print("Training step took {} seconds".format(time.time() - train_start_time))
print("Best dev acc {0}".format(best_per))
print("")

# remember to eval after loading the model. for the reason of batchnorm and dropout
cur_time = time.time()
f1 = get_f1(best_model, "test")
print("Test step took {} seconds".format(time.time() - cur_time))

serial_number = str(random.randint(0,248))
this_model_path = config.model_path + "_" + serial_number
print("Dumping model to {0}".format(this_model_path))
torch.save(best_model.state_dict(), this_model_path)
