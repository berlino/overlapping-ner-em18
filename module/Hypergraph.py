"""
Pytorch implemenation of Hypergraph
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from torch.autograd import Variable
from collections import defaultdict
import pdb


def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # vec is only 1d vec
    # return the argmax as a python int
    _, idx = torch.max(vec, 0)
    return to_scalar(idx)

def create_empty_var(if_gpu):
    if if_gpu:
        loss = Variable(torch.Tensor([0]).cuda())
    else:
        loss = Variable(torch.Tensor([0])) 
    return loss

def log_sum_exp(vec_list):
    """
    Compute log sum exp in a numerically stable way for the forward algorithm
    vec is n * m, norm in row
    return n * 1
    """
    if type(vec_list) == list:
        mat = torch.stack(vec_list, 1)
    else:
        mat = vec_list
    row, column = mat.size()
    ret_l = []
    for i in range(row):
        vec = mat[i]
        max_score = vec[argmax(vec)]
        max_score_broadcast = max_score.view(-1).expand(1, vec.size()[0])
        ret_l.append(max_score + \
                     torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
    return torch.cat(ret_l, 0)

def log_sum_exp_t(vec_A, vec_B, vec_C):
    """
    vec_size: batch_size * label_size
    """
    batch_size, label_size = vec_A.size()
    vec_A = vec_A.view(-1, 1).squeeze(1)
    vec_B = vec_B.view(-1, 1).squeeze(1)
    vec_C = vec_C.view(-1, 1).squeeze(1)

    vec_D = log_sum_exp([vec_A, vec_B, vec_C])
    return vec_D.view(batch_size, label_size)

def log_sum_exp_b(vec_A, vec_B):
    """
    vec_size: batch_size * label_size
    """
    batch_size, label_size = vec_A.size()
    vec_A = vec_A.view(-1, 1).squeeze(1)
    vec_B = vec_B.view(-1, 1).squeeze(1)

    vec_D = log_sum_exp([vec_A, vec_B])
    return vec_D.view(batch_size, label_size)


class Hypergraph(nn.Module):
    """
    Naive implementaion would be every instance is paired with partial tree structure. 
    But padding would be a problme then.
    For consideration of efficiency, hypergraph is computed in batch mode.
    The representation of hypergraph is A, E, T, I, X 
    """

    def __init__(self,
                 config) -> None:
        super(Hypergraph, self).__init__()
        self.config = config
        self.label_size = config.label_size
        self.C = config.C
        self.hidden_size = config.semi_hidden_size
        self.if_margin = config.if_margin
        self.beta = config.beta

        self.II_lin = nn.Linear(config.f_hidden_size * 4, self.label_size)
        self.TX_lin = nn.Linear(config.f_hidden_size * 2, self.label_size)
        self.IX_lin = nn.Linear(self.hidden_size * 2, self.label_size)
        self.TI_lin = nn.Linear(config.f_hidden_size * 2, self.label_size)


        self.f_cell = nn.LSTMCell(config.f_hidden_size * 2, config.semi_hidden_size)
        self.b_cell = nn.LSTMCell(config.f_hidden_size * 2, config.semi_hidden_size)

    def _gen_TX_batch(self, inputs, entity_batch):
        batch_size, sent_len, feat_dim = inputs.size()
        null_batch = torch.ones(batch_size, sent_len, self.label_size)
        for i in range(batch_size):
            for start, end, label in entity_batch[i]:
                null_batch[i, start, label] = 0
        ret_var = Variable(null_batch)
        if self.config.if_gpu:
            ret_var = ret_var.cuda()
        return ret_var

    def _gen_II_batch(self, inputs, entity_batch):
        """
        only overlapping mentions with the same type have this kind of feature
        """
        batch_size, sent_len, feat_dim = inputs.size()
        if sent_len == 1:
            return None
        II_batch = torch.zeros(batch_size, sent_len - 1, self.label_size)
        for i in range(batch_size):
            start_dic = defaultdict(list)
            for start, end, label in entity_batch[i]:
                start_dic[(start, label)].append(end)
            
            for (start, label), end_list in start_dic.items():
                if len(end_list) > 0:
                    min_i = min(end_list)
                    max_i = max(end_list)
                    for j in range(start, max_i):
                        # It could be more than 1
                        II_batch[i, j, label] += 1
        ret_var = Variable(II_batch)
        if self.config.if_gpu:
            ret_var = ret_var.cuda()
        return ret_var

    def _gen_TI_batch(self, inputs, entity_batch):
        batch_size, sent_len, feat_dim = inputs.size()
        TI_batch = torch.zeros(batch_size, sent_len, self.label_size)
        for i in range(batch_size):
            start_dic = defaultdict(list)
            for start, end, label in entity_batch[i]:
                start_dic[(start, label)].append(end)
            
            for (start, label), end_list in start_dic.items():
                TI_batch[i, start, label] = 1
        ret_var = Variable(TI_batch)
        if self.config.if_gpu:
            ret_var = ret_var.cuda()
        return ret_var

    def _filter_entity(self, inputs, entity_batch):
        batch_size, sent_len, feat_dim = inputs.size()
        ret_l = []
        for i in range(batch_size):
            sent_ents = []
            for start, end, label in entity_batch[i]:
                if end + 1 - start <= self.C:
                    sent_ents.append((start, end, label))
            ret_l.append(sent_ents)
        return ret_l

    def _marginize(self, TI_scores, TX_scores, entity_batch):
        """
        Add softmax margin
        """
        batch_size, sent_len, label_size = TI_scores.size()

        FP_mat = torch.zeros(batch_size, sent_len, label_size)
        FN_mat = torch.zeros(batch_size, sent_len, label_size)

        for b in range(batch_size):
            for i in range(sent_len):
                for k in range(label_size):
                    bool_TX = True
                    for j in range(i, sent_len):
                        if (i, j, k) in entity_batch[b]:
                            bool_TX  = False
                    if bool_TX:
                        FP_mat[b,i,k] = 1
                    else:
                        FN_mat[b,i,k] = self.beta
        FP_var = Variable(FP_mat)
        FN_var = Variable(FN_mat)
        if self.config.if_gpu:  
            FP_var = FP_var.cuda()
            FN_var = FN_var.cuda()
        TI_scores = TI_scores + FP_var
        TX_scores = TX_scores + FN_var
        return TI_scores, TX_scores

    def forward(self,
                inputs: torch.FloatTensor,
                entity_batch: List) -> Dict[str, torch.Tensor]:
        """
        inputs: matrix with size: batch_size * sent_len * feat_dim
        entities: list of (start, end, label)

        Output dictionary contains:
        expectation: the expected value of partition function in log space
        loss: if chunks is given
        """
        output = {}
        batch_size, sent_len, feat_dim = inputs.size()

        if self.config.if_C:
            entity_batch = self._filter_entity(inputs, entity_batch)

        TX_scores = self.TX_lin(inputs)
        TX_batch_mask = self._gen_TX_batch(inputs, entity_batch)
        TX_ner_scores = torch.mul(TX_scores, TX_batch_mask).sum(1).sum(1)
        ner_scores = TX_ner_scores

        span_vectors = self._gen_seg_mat(inputs)
        IX_scores = self.IX_lin(span_vectors)
        IX_ner_scores = self.score_chunk(IX_scores, entity_batch)
        ner_scores = ner_scores + IX_ner_scores

        TI_scores = self.TI_lin(inputs)
        TI_batch_mask = self._gen_TI_batch(inputs, entity_batch)
        TI_ner_scores = torch.mul(TI_scores, TI_batch_mask).sum(1).sum(1)
        ner_scores = ner_scores + TI_ner_scores

        if sent_len > 1:
            II_list = []
            for pos in range(sent_len - 1):
                II_pos = self.II_lin(torch.cat([inputs[:, pos], inputs[:, pos + 1]], 1))  # batch_size * 1
                II_list.append(II_pos)
            II_scores = torch.stack(II_list, 1)  #  batch_size * sent_len - 1 * label_size
            II_batch_mask = self._gen_II_batch(inputs, entity_batch)
            II_ner_scores = torch.mul(II_scores, II_batch_mask).sum(1).sum(1)
            ner_scores = ner_scores + II_ner_scores
        else:
            II_scores = None

        #TODO: TI constrain
        if self.if_margin:
            TI_scores, TX_scores = self._marginize(TI_scores, TX_scores, entity_batch)
        TX_scores = TX_scores.transpose(0, 1).contiguous()
        if II_scores is not None:
            II_scores = II_scores.transpose(0, 1).contiguous()
        if TI_scores is not None:
            TI_scores = TI_scores.transpose(0, 1).contiguous()
        inside_score = self.inside(IX_scores, TX_scores, II_scores, TI_scores)

        assert inside_score.size() == ner_scores.size()
        output["expectation"] = inside_score
        # negative log-likelihood
        loss_vec = (inside_score - ner_scores)
        output["loss"] = loss_vec.mean()

        # loss could be negative if we restrict the max length
        loss_vec_relu = F.relu(loss_vec)
        diff = (loss_vec_relu - loss_vec).max().cpu().data[0]
        # if diff > 1e-4: # arould singel precision
        if output["loss"].cpu().data[0] < 0:
            # pdb.set_trace()
            output["loss"] = loss_vec_relu.mean()
            print("Nega loss! diff {0} with length {1}".format(diff, sent_len))
        return output  

    def inside(self,
               IX_scores: torch.FloatTensor,
               TX_scores: torch.FloatTensor,
               II_scores: torch.FloatTensor,
               TI_scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        :param  
        IX_scores: sent_len * sent_len * batch_size * label_size
        TX_scores:  sent_len * batch_size * label_size
        II_scores: sent_len - 1 * batch_size * label_size
        :return: Z
        """
        sent_len, sent_len, batch_size, label_size = IX_scores.size() 
        if sent_len == 1:
            if TI_scores is None:
                return log_sum_exp_b(IX_scores[0,0], TX_scores[0]).sum(1)
            else:
                return log_sum_exp_b(IX_scores[0,0] + TI_scores[0], TX_scores[0]).sum(1)


        score_list = []
        for i in range(sent_len):
            if self.config.if_C:
                outlier = min(i + self.C - 1, sent_len - 1)
            else:
                outlier = sent_len - 1

            pre_vec = IX_scores[i, outlier]
            for j in reversed(range(i, outlier)):
                # pdb.set_trace()
                vec_A = pre_vec + II_scores[j]
                vec_B = IX_scores[i, j]
                vec_C = vec_A + vec_B
                # merged_ = log_sum_exp_t(vec_A, vec_B, vec_C)
                merged_v = torch.stack([vec_A, vec_B, vec_C], 2)
                merged = (merged_v - F.log_softmax(merged_v, dim=2)).mean(2)
                pre_vec = merged

            ent_vec = pre_vec
            ent_vec = ent_vec + TI_scores[i]
            # final_vec_ = log_sum_exp_b(ent_vec, null_scores[i])
            final_vec = torch.stack([ent_vec, TX_scores[i]], 2) # batch_size * label_size * 2
            merged_final_vec = (final_vec - F.log_softmax(final_vec, dim=2)).mean(2) # batch_size * label_size
            # pdb.set_trace()
            score_list.append(merged_final_vec.sum(1))  # batch_size
        
        score_mat = torch.stack(score_list, 1) # batch_size * sent_len
        overall_score = score_mat.sum(1)
        # overall_score = sum(score_list)
        return overall_score

    def outside(self,
                inputs: torch.FloatTensor) -> torch.FloatTensor:
        """
        for sanity check
        """
        pass

    def score_chunk(self,
                    IX_scores: torch.FloatTensor,
                    entity_batch: List) -> torch.Tensor:
        sent_len, sent_len, batch_size, label_size = IX_scores.size() 
        gold_score = []

        for i in range(batch_size):
            entities = entity_batch[i]
            
            a_score = create_empty_var(self.config.if_gpu)
            for start, end, label in entities:
                a_score += IX_scores[start, end, i, label]
            gold_score.append(a_score)
        
        return torch.cat(gold_score, 0)

    def decode(self,
               inputs: torch.FloatTensor) -> List:
        """
        :param inputs: matrix of embeddings
        :return: partial overlapping structure

        The main purpose of decoding is not for metrics, but for interpretation
        """
        batch_size, sent_len, feat_dim = inputs.size()

        II_list = []
        for pos in range(sent_len - 1):
            II_pos = self.II_lin(torch.cat([inputs[:, pos], inputs[:, pos + 1]], 1))  # batch_size * 1
            II_list.append(II_pos)
        if len(II_list) != 0:
            II_scores = torch.stack(II_list, 0)  #  sent_len - 1 * batch_size * label_size

        IX_vectors = self._gen_seg_mat(inputs) # sent_len * sent_len * batch_size * feat_dim
        IX_scores = self.IX_lin(IX_vectors)
        TX_scores = self.TX_lin(inputs)  # batch_size * sent_len * feat_dim
        TI_scores = self.TI_lin(inputs)

        ret_list = []
        for i in range(batch_size):
            entity_list = []
            for j in range(sent_len):
                for k in range(self.config.label_size):
                    if self.config.if_C:
                        outlier = min(j + self.C - 1, sent_len - 1)
                    else:
                        outlier = sent_len - 1

                    def recur_find(s):
                        if s == outlier:
                            return IX_scores[j, s, i, k], [s]
                        best_value, best_ends = recur_find(s+1)

                        v_A = IX_scores[j, s, i, k]
                        v_B = best_value + II_scores[s, i, k]
                        if v_A.data[0] > 0 and v_B.data[0] > 0:
                            best_ends.append(s)
                            return v_A + v_B, best_ends
                        elif v_B.data[0] <= 0 and v_A.data[0] >= v_B.data[0]:
                            return v_A, [s]
                        else:
                            return v_B, best_ends
                    
                    seq_score, end_list = recur_find(j)
                    seq_score = seq_score + TI_scores[i, j, k]
                    if TX_scores[i, j, k].data[0] < seq_score.data[0]:
                        for end in end_list:
                            entity_list.append((j, end, k))
            ret_list.append(entity_list)
        return ret_list


    def _gen_seg_mat(self, feat):
        """
        generate the span representation 
        :param feat:  batch_size * sent_len * input_feat_size
        :return: sen_len * sent_len * batch_size * output_hidden_size
        """
        batch_size, sent_len, feat_size = feat.size()
        # contiguous create a new storage, so don't mind about the original storage
        feat = feat.transpose(0, 1).contiguous()
        # diagonal element
        init_f_cell_state = torch.rand(batch_size, self.hidden_size)
        init_b_cell_state = torch.rand(batch_size, self.hidden_size)
        init_f_hidden_state = torch.rand(batch_size, self.hidden_size)
        init_b_hidden_state = torch.rand(batch_size, self.hidden_size)
        init_f_cell_state = Variable(init_f_cell_state)
        init_b_cell_state = Variable(init_b_cell_state)
        init_f_hidden_state = Variable(init_f_hidden_state)
        init_b_hidden_state = Variable(init_b_hidden_state)
        if self.config.if_gpu:
            init_f_cell_state = init_f_cell_state.cuda()
            init_b_cell_state = init_b_cell_state.cuda()
            init_f_hidden_state = init_f_hidden_state.cuda()
            init_b_hidden_state = init_b_hidden_state.cuda()

        # v[i][j] i > j ==> forward  i < j ==> backward
        f_span_dic = {}
        b_span_dic = {}
        f_cell_dic = {}
        b_cell_dic = {}
        for i in range(sent_len):
            f_span_dic[(i,i)], f_cell_dic[(i,i)] = self.f_cell(feat[i], (init_f_hidden_state, init_f_cell_state))
            b_span_dic[(i,i)], b_cell_dic[(i,i)] = self.b_cell(feat[i], (init_b_hidden_state, init_b_cell_state))
            for j in range(i + 1, sent_len):
                f_span_dic[(i,j)], f_cell_dic[(i,j)] = self.f_cell(feat[j], (f_span_dic[(i, j-1)], f_cell_dic[(i,j-1)]))
            for j in reversed(range(0, i)):
                b_span_dic[(i,j)], b_cell_dic[(i,j)] = self.b_cell(feat[j], (b_span_dic[(i, j+1)], b_cell_dic[(i,j+1)]))
        # only i >= j is valid representation
        span_list = []
        null_element = Variable(torch.Tensor(batch_size, self.hidden_size*2).zero_())
        if self.config.if_gpu:
            null_element = null_element.cuda()
        for i in range(sent_len):
            cache_list = []
            for j in range(i, sent_len):
                cache_list.append(torch.cat([f_span_dic[i,j], b_span_dic[j,i]], 1))
            cache_list = [null_element] * i + cache_list
            span_list.append(torch.stack(cache_list, 0))
        span_vectors = torch.stack(span_list, 0)
        # pdb.set_trace()
        return span_vectors
