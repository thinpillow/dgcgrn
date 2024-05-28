import random
import math
import numpy as np
import pandas as pd
import re
import scipy.sparse as sp
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from cvae_model import cvae_models, cvae_pretrain
import warnings
import matplotlib.pyplot as plt

from pytorchtools import EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn import metrics, preprocessing
from torch_geometric.data import Data
from dgcn.dgcn import DGCN_link_prediction
from torch_geometric_signed_directed.utils import in_out_degree, directed_features_in_out
from torch_geometric_signed_directed import DirectedData

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)  # 1e-2=10^-2
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--concat', type=int, default=1)
parser.add_argument('--topn', type=int, default=15)
args = parser.parse_args()

epochs = range(1, 101)

def kmer_func(seq):
    X = [None] * len(seq)
    for i in range(len(seq)):
        a = seq[i]
        t = 0
        l = []
        for index in range(len(a)):
            t = a[index:index + 5]
            if (len(t)) == 5:
                l.append(t)
        X[i] = l
    return np.array(X)

def encode_matrix(seq_matrix):
    seq_X = [None] * len(seq_matrix)
    for j in range(len(seq_matrix)):
        data = seq_matrix[j]
        ind_to_char = k_mer
        char_to_int = dict((c, i) for i, c in enumerate(ind_to_char))  # 枚举
        integer_encoded = [char_to_int[char] for char in data]
        onthot_encoded = list()
        for value in integer_encoded:
            letter = [0 for _ in range(len(ind_to_char))]
            letter[value] = 1
            onthot_encoded.append(letter)
        seq_X[j] = onthot_encoded
    return seq_X

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

cvae_model = cvae_models.VAE(encoder_layer_sizes=[66, 100],
                             latent_size=8,
                             decoder_layer_sizes=[100, 60])

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
        X_list.append(augmented_features)
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list

result = []
all_auc = []
all_val_auc = []
all_test_auc = []
pathY = ''
pathts = ''

cold_ts = pd.read_csv(pathts)
gene = cold_ts.columns
gene = np.array(gene)
ID = [i for i in range(1, 1484 + 1)]
genesID = list(zip(gene, ID))

integrated_gold_network = pd.read_csv(pathY)
integrated_gold_network = np.array(integrated_gold_network)
rowNumber = []
colNumber = []
    for i in range(len(integrated_gold_network)):
        genes1 = integrated_gold_network[i][0]
        for j in range(len(genesID)):
            if genes1 == genesID[j][0]:
                rownum = genesID[j][1]
                rowNumber.append(rownum)

        genes2 = integrated_gold_network[i][1]
        for k in range(len(genesID)):
            if genes2 == genesID[k][0]:
                colnum = genesID[k][1]
                colNumber.append(colnum)
    geneNetwork = np.zeros((1484, 1484))
    for i in range(len(rowNumber)):
        r = rowNumber[i] - 1
        c = colNumber[i] - 1
        geneNetwork[r][c] = 1
    adj_data = []
    for i in range(1484):
        for j in range(1484):
            if geneNetwork[i][j] == 1:
                adj_data.append(int(geneNetwork[i][j]))
    adj_data = np.array(adj_data)
    adj_shape = np.array(geneNetwork.shape)
    adj_indices = sp.csr_matrix(geneNetwork).indices
    adj_indptr = sp.csr_matrix(geneNetwork).indptr

    features_matrix = pd.read_csv()
    features_matrix = np.transpose(features_matrix)
    features_matrix = np.array(features_matrix)
    features = []
    for i in range(features_matrix.shape[0]):
        for j in range(features_matrix.shape[1]):
            features.append(features_matrix[i][j])
    x_list = get_augmented_features(args.concat)
    for i in range(len(x_list)):
        X_list = x_list[i]

    original_data = pd.read_csv()
    original_data = np.array(original_data)
    seq_data = original_data[:, 2]
    seq_feature = []
    for i in range(len(seq_data)):
        seq = [seq_data[i]]
        ad = len(seq[0])
        if len(seq[0]) > 1:
            k_mer = kmer_func(seq)
            k_mer = np.unique(k_mer)
            k_mers = []
            k_mers.append(k_mer)
            onehot = encode_matrix(k_mers)
            onehot = np.array(onehot)
            onehot = onehot.astype('float32')

            onehot = torch.tensor(onehot)
            rnn3 = nn.GRU(onehot.shape[2], 10, 1, bidirectional=True)

            h1 = torch.randn(2, onehot.shape[2], 10)
            c1 = torch.randn(2, onehot.shape[2], 10)

            output3, hn3 = rnn3(onehot, h1)
            bi_gru = output3[-1]
            bi_gru1 = bi_gru.detach().numpy()
            bi_gru_avg = bi_gru1.mean(axis=0)
            seq_feature.append(bi_gru_avg)
        else:
            non_geng = np.zeros(20)
            for j in range(len(non_geng)):
                non_geng[j] = math.pow(10, -99)
            seq_feature.append(non_geng)
    seq_feature = np.array(seq_feature)
    X_list = np.hstack((X_list, seq_feature))
    bio_data = pd.read_csv()
    bio_data = np.array(bio_data)
    X_list = np.hstack((X_list, bio_data))

    x1_list = []
    for i in range(X_list.shape[0]):
        for j in range(X_list.shape[1]):
            x1_list.append(X_list[i][j])
    attr_data = np.array(x1_list)
    attr_shape = np.array(X_list.shape)
    attr_indices = sp.csr_matrix(X_list).indices
    attr_indptr = sp.csr_matrix(X_list).indptr

    adj = sp.csr_matrix((adj_data, adj_indices,
                         adj_indptr), shape=adj_shape)
    features = sp.csr_matrix((attr_data, attr_indices,
                              attr_indptr), shape=attr_shape)
    coo = adj.tocoo()
    values = torch.from_numpy(coo.data)
    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    features = torch.from_numpy(features.todense()).float()
    data = Data(x=features, edge_index=indices, edge_weight=values)
    if hasattr(data, 'edge_weight'):
        edge_weight = data.edge_weight
    else:
        edge_weight = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = DirectedData(
        edge_index=data.edge_index, edge_weight=edge_weight, init_data=data).to(device)

    model = DGCN_link_prediction(num_features=X_list.shape[1], hidden=16, label_dim=2, dropout=0.6).to(device)
    criterion = nn.CrossEntropyLoss()


    def train(X, y, edge_index, edge_in, in_weight,
              edge_out, out_weight, query_edges):
        model.train()
        out = model(X, edge_index, edge_in=edge_in, in_w=in_weight,
                    edge_out=edge_out, out_w=out_weight, query_edges=query_edges)
        out = F.softmax(out, dim=1)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
        fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), out.max(dim=1)[0].cpu().detach().numpy())
        train_auc = np.trapz(tpr, fpr)
        y_score = out.max(dim=1)[0]
        max_num = max(y_score)
        min_num = min(y_score)
        for i in range(len(y_score)):
            y_score[i] = (y_score[i] - min_num) / (max_num - min_num)
        new_edge_weight = y_score
        return loss.detach().item(), train_acc, train_auc, new_edge_weight
                  
    def test(X, y, edge_index, edge_in, in_weight,
             edge_out, out_weight, query_edges):
        model.eval()
        with torch.no_grad():
            out = model(X, edge_index, edge_in=edge_in, in_w=in_weight,
                        edge_out=edge_out, out_w=out_weight, query_edges=query_edges)
        out = F.softmax(out, dim=1)
        test_acc = metrics.accuracy_score(y.cpu(), out.max(dim=1)[1].cpu())
        fpr, tpr, thresholds = metrics.roc_curve(y.cpu(), out.max(dim=1)[0].cpu().detach().numpy())
        test_auc = np.trapz(tpr, fpr)
        precision_aupr, recall_aupr, _ = metrics.precision_recall_curve(y.cpu(), out.max(dim=1)[0].cpu())
        aupr = metrics.auc(recall_aupr, precision_aupr)
        f1 = metrics.f1_score(y.cpu(), out.max(dim=1)[1].cpu())
        return test_acc, test_auc, f1, aupr
