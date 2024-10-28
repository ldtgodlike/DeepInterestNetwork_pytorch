import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class Attention(nn.Module):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.hidden_units = hidden_units
        self.linear1 = nn.Linear(hidden_units * 4, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, 1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, queries, keys, keys_length):
        """
        queries:     [B, H]
        keys:        [B, T, H]
        keys_length: [B, len]
        """
        B, H = queries.shape
        T = keys.shape[1]

        # Tile queries
        queries = queries.unsqueeze(1).repeat(1, T, 1)  # B, T, H

        # Concatenate queries, keys, queries-keys, queries*keys
        x = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)  # B, T, 4H

        # Dense layers
        x = self.sigmoid1(self.linear1(x))  # B, T, 80
        x = self.sigmoid2(self.linear2(x))  # B, T, 40
        x = self.linear3(x)  # B, T, 1
        x = x.view(B, 1, T)  # B, 1, T

        # Masking
        key_masks = (torch.arange(T).to(x.device).expand(B, T) < keys_length.unsqueeze(1)).unsqueeze(1)  # B, 1, T
        paddings = torch.ones_like(x).to(x.device) * (-2 ** 32 + 1)  # B, 1, T
        x = torch.where(key_masks, x, paddings)  # B, 1, T

        # Scale
        x = x / (self.hidden_units ** 0.5)  # B, 1, T

        # Softmax
        x = F.softmax(x, dim=-1)  # B, 1, T
        # Weighted sum
        #                      key: B, T, H
        x = torch.matmul(x, keys).view(B, H)  # B, H
        return x


class AttentionMultiItems(nn.Module):
    def __init__(self, hidden_units):
        super(AttentionMultiItems, self).__init__()
        self.hidden_units = hidden_units
        self.linear1 = nn.Linear(hidden_units * 4, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, 1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()

    def forward(self, queries, keys, keys_length):
        """
        queries:     [B, N, H]
        keys:        [B, T, H]
        keys_length: [B, len]
        """
        B, N, H = queries.shape
        T = keys.shape[1]

        # Tile queries
        queries = queries.unsqueeze(2).repeat(1, 1, T, 1)  # B, N, T, H

        # Tile keys
        keys = keys.unsqueeze(1).repeat(1, N, 1, 1)  # B, N, T, H

        # Concatenate queries, keys, queries-keys, queries*keys
        x = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)  # B, N, T, 4H

        # Dense layers
        x = self.sigmoid1(self.linear1(x))  # B, N, T, 80
        x = self.sigmoid2(self.linear2(x))  # B, N, T, 40
        x = self.linear3(x)  # B, N, T, 1
        x = x.view(B, N, 1, T)
        # Masking
        key_masks = (torch.arange(T).to(x.device).unsqueeze(0).repeat(B, 1) < keys_length.unsqueeze(1)).unsqueeze(1).repeat(1, N, 1).unsqueeze(
            -2)  # B, N, 1, T
        paddings = torch.ones_like(x).to(x.device) * (-2 ** 32 + 1)  # B, N, 1, T
        x = torch.where(key_masks, x, paddings)  # B, N, 1, T
        # Scale
        x = x / (self.hidden_units ** 0.5)  # B, N, 1, T

        # Softmax
        x = F.softmax(x, dim=-1)  # B, N, T
        # Weighted sum                                                                                 # key: B, N, H
        x = torch.matmul(x, keys).view(B, N, H)  # B, N, H

        return x


# class EmbeddingLayer(nn.Module):
#     def __init__(self, user_count, item_count, cate_count, cate_list, hidden_units):
#         super(EmbeddingLayer, self).__init__()
#         self.user_emb = nn.Embedding(user_count, hidden_units)
#         self.item_emb = nn.Embedding(item_count, hidden_units // 2)
#         self.cate_emb = nn.Embedding(cate_count, hidden_units // 2)
#         cate_list = torch.tensor(cate_list, dtype=torch.long).cuda()
#
#     def forward(self, u, i, j, hist_i):
#         # User embedding
#         u_emb = self.user_emb(u)
#
#         # Item embedding
#         ic = cate_list[i]
#         i_emb = torch.cat([self.item_emb(i), self.cate_emb(ic)], dim=1)
#
#         # Negative item embedding
#         jc = cate_list[j]
#         j_emb = torch.cat([self.item_emb(j), self.cate_emb(jc)], dim=1)
#
#         # History item embedding
#         hc = cate_list[hist_i]
#         h_emb = torch.cat([self.item_emb(hist_i), self.cate_emb(hc)], dim=2)
#
#         return u_emb, i_emb, j_emb, h_emb


class AttentionLayer(nn.Module):
    def __init__(self, hidden_units, muilt=False):
        super(AttentionLayer, self).__init__()
        if muilt:
            self.attention = AttentionMultiItems(hidden_units)
        else:
            self.attention = Attention(hidden_units)

    def forward(self, i_emb, h_emb, sl):
        hist_i = self.attention(i_emb, h_emb, sl)
        return hist_i


class LinearBlock(nn.Module):
    def __init__(self, intput_dim, hidden_units):
        super(LinearBlock, self).__init__()
        self.batch_normal = nn.BatchNorm1d(hidden_units)
        self.linear = nn.Linear(intput_dim, hidden_units)
        self.hidden_units = hidden_units

    def forward(self, hist_i):
        hist_i = self.batch_normal(hist_i)
        hist_i = hist_i.view(-1, self.hidden_units)
        hist_i = self.linear(hist_i)
        return hist_i

class LinearBlockSub(nn.Module):
    def __init__(self, intput_dim, hidden_units):
        super(LinearBlockSub, self).__init__()
        self.batch_normal = nn.BatchNorm1d(hidden_units)
        self.linear = nn.Linear(intput_dim, hidden_units)
        self.hidden_units = hidden_units

    def forward(self, hist_i):
        hist_i = hist_i.view(-1, self.hidden_units)
        hist_i = self.batch_normal(hist_i)
        hist_i = self.linear(hist_i)
        return hist_i


class FCNBlock(nn.Module):
    def __init__(self, hidden_units):
        super(FCNBlock, self).__init__()
        self.batch_normal = nn.BatchNorm1d(hidden_units * 3)
        self.linear1 = nn.Linear(hidden_units * 3, 80)
        self.linear2 = nn.Linear(80, 40)
        self.linear3 = nn.Linear(40, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, din_i):
        din_i = self.batch_normal(din_i)
        din_i = self.sigmoid1(self.linear1(din_i))
        din_i = self.sigmoid2(self.linear2(din_i))
        din_i = self.linear3(din_i)
        return din_i


class DIN(nn.Module):
    def __init__(self, user_count, item_count, cate_count, predict_ads_num,
                 hidden_units=128):
        super(DIN, self).__init__()
        #self.embedding_layer = EmbeddingLayer(user_count, item_count, cate_count, cate_list, hidden_units)
        # self.user_emb_w = nn.Parameter(torch.Tensor(user_count, hidden_units))
        self.item_emb_w = nn.Parameter(torch.Tensor(item_count, hidden_units // 2))
        self.item_b = nn.Parameter(torch.zeros(item_count))
        self.cate_emb_w = nn.Parameter(torch.Tensor(cate_count, hidden_units // 2))

        self.attention_layer_i = AttentionLayer(hidden_units, muilt=False)
        self.attention_layer_j = AttentionLayer(hidden_units, muilt=False)
        self.linearBlock_i = LinearBlock(hidden_units, hidden_units)
        self.linearBlock_j = LinearBlock(hidden_units, hidden_units)
        self.linearBlock_sub = LinearBlockSub(hidden_units, hidden_units)
        self.fcn1 = FCNBlock(hidden_units)
        self.fcn2 = FCNBlock(hidden_units)

        self.attention_layer_sub = AttentionLayer(hidden_units, muilt=True)
        self.fcn_sub = FCNBlock(hidden_units)

        #self.predict_batch_size = predict_batch_size
        self.predict_ads_num = predict_ads_num
        #cate_list = torch.Tensor(cate_list).long().cuda()
        self.hidden_units = hidden_units

        self.reset_parameters()

    def reset_parameters(self):
        #self.user_emb_w.data.normal_(0,1)
        self.item_emb_w.data.normal_(0,1)
        self.cate_emb_w.data.normal_(0,1)

    def forward(self, iids, y, hist_i, sl, cate_list):
        # Embedding layer
        #u_emb, i_emb, j_emb, h_emb = self.embedding_layer(u, i, y, hist_i)
        ic = torch.index_select(cate_list, 0, iids)
        i_emb = torch.cat([F.embedding(iids, self.item_emb_w), F.embedding(ic, self.cate_emb_w)], dim=1)
        i_b = torch.index_select(self.item_b, 0, iids)

        jc = torch.index_select(cate_list, 0, y)
        j_emb = torch.cat([F.embedding(y, self.item_emb_w), F.embedding(jc, self.cate_emb_w)], dim=1)
        j_b = torch.index_select(self.item_b, 0, y)

        hc = torch.index_select(cate_list, 0, hist_i.view(-1))
        h_emb = torch.cat([F.embedding(hist_i.view(-1), self.item_emb_w), F.embedding(hc, self.cate_emb_w)], dim=1).view(iids.shape[0], -1,self.hidden_units)

        # Attention layer for positive item
        hist_i = self.attention_layer_i(i_emb, h_emb, sl)
        u_emb_i = self.linearBlock_i(hist_i)

        # Attention layer for positive item
        hist_j = self.attention_layer_j(j_emb, h_emb, sl)
        u_emb_j = self.linearBlock_j(hist_j)

        # FC layers for positive item
        din_i = torch.cat([u_emb_i, i_emb, u_emb_i * i_emb], dim=-1)
        din_i = self.fcn1(din_i)
        din_i = din_i.view(-1)

        # FC layers for negative item
        din_j = torch.cat([u_emb_j, j_emb, u_emb_j * j_emb], dim=-1)
        din_j = self.fcn1(din_j)
        din_j = din_j.view(-1)

        # Compute logits
        x = i_b - j_b + din_i - din_j
        logits = i_b + din_i

        # Prediction for selected items
        item_emb_all = torch.cat([self.item_emb_w, F.embedding(cate_list, self.cate_emb_w)], dim=1)
        item_emb_sub = item_emb_all[:self.predict_ads_num, :]
        item_emb_sub = item_emb_sub.unsqueeze(0).repeat(iids.shape[0], 1, 1)
        hist_sub = self.attention_layer_sub(item_emb_sub, h_emb, sl)
        u_emb_sub = self.linearBlock_sub(hist_sub)

        item_emb_sub = item_emb_sub.view(-1, self.hidden_units)
        din_sub = torch.cat([u_emb_sub, item_emb_sub, u_emb_sub * item_emb_sub], dim=-1)
        din_sub = self.fcn_sub(din_sub)
        din_sub = din_sub.view(-1, self.predict_ads_num)

        logits_sub = torch.sigmoid(i_emb[:, :self.predict_ads_num] + din_sub)
        logits_sub = logits_sub.view(-1, self.predict_ads_num, 1)

        # Compute loss
        #loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1).float(), reduction='mean')
        logits = torch.sigmoid(logits.view(-1))

        return logits, logits_sub, x, i_b, j_b, din_i, din_j

    def eval_step(self, x, i_b, j_b, din_i, din_j):
        with torch.no_grad():
            nf_auc = torch.mean((x > 0).float())
            score_i = torch.sigmoid(i_b + din_i).view(-1, 1)
            score_j = torch.sigmoid(j_b + din_j).view(-1, 1)
            p_and_n = torch.cat([score_i, score_j], dim=-1)
        return nf_auc, p_and_n


if __name__ == "__main__":
    # Test the Attention layer
    B = 2
    T = 5
    N = 3
    H = 10
    hidden_units = H
    queries = torch.randn(B, N, hidden_units)
    keys = torch.randn(B, T, hidden_units)
    keys_length = torch.ones((B, N, 1)) * 4

    attention_layer = AttentionMultiItems(hidden_units)
    outputs = attention_layer(queries, keys, keys_length)
    print(outputs.shape)

    # class DIN(nn.Module):
    #     def __init__(self):
