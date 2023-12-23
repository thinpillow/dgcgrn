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
early_stopping = EarlyStopping(patience=7, min_delta=0)
for net in range(3):
    network = ['cold', 'heat', 'oxidativestress']
    # network = ['oxidativestress']
    print("=================the " + network[net] + " network=================")
    pathY = r'D:\data\bulkGRN\Ecoli\integrated_gold_network.tsv'
    pathts = r'D:\ecoli\\' + str(network[net]) + '_time_3_replice_log2.tsv'

    cold_ts = pd.read_csv(pathts, sep='\t')
    gene = cold_ts.columns
    gene = np.array(gene)
    ID = [i for i in range(1, 1484 + 1)]
    genesID = list(zip(gene, ID))

    integrated_gold_network = pd.read_csv(pathY, sep='\t', header=None)
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

    features_matrix = pd.read_csv(
        r'D:\ecoli\\' + str(network[net]) + '_time_3_replice_log2.tsv', sep='\t')
    features_matrix = np.transpose(features_matrix)
    features_matrix = np.array(features_matrix)
    features = []
    for i in range(features_matrix.shape[0]):
        for j in range(features_matrix.shape[1]):
            features.append(features_matrix[i][j])
    x_list = get_augmented_features(args.concat)
    for i in range(len(x_list)):
        X_list = x_list[i]

    original_data = pd.read_csv(r'D:\data\bulkGRN\Ecoli\GENE1461infor_feature4.csv', sep=',')
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
    bio_data = pd.read_csv(r'D:\GENE1461infor_feature.csv', sep=',', header=None)
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

    model = DGCN_link_prediction(
        num_features=X_list.shape[1], hidden=16, label_dim=2, dropout=0.6).to(device)
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


    kf = KFold(n_splits=5, shuffle=True)
    list1 = []
    list2 = []
    list0 = []
    all_data = []
    for i in range(1484):
        for j in range(1484):
            if geneNetwork[i][j] == 1:
                list1.append([i, j])
            else:
                list2.append([i, j])
    list0 = list1 + list2
    all_data = list1 + random.sample(list2, len(adj_data))
    random.shuffle(all_data)
    all_data = np.array(all_data)
    sum1 = 0
    locals()[f'best_auc_{net}'] = []
    locals()[f'best_aupr_{net}'] = []
    locals()[f'best_f1_{net}'] = []
    locals()[f'best_acc_{net}'] = []
    for ki in range(5):
        sum2 = 0
        print('===================第{}次五折交叉====================='.format(ki + 1))
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for train_index, test_index in kf.split(all_data):
            train_data, test_data = all_data[train_index], all_data[test_index]
            train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=1, shuffle=True)
            labels = {}
            train_labels = []
            val_labels = []
            test_labels = []
            list00 = []
            list11 = []
            for a in range(len(list0)):
                list00.append(tuple(list0[a]))
            for b in range(len(list1)):
                list11.append(tuple(list1[b]))
            for i in range(len(list0)):
                if list00[i] in list11:
                    labels[list00[i]] = 1
                else:
                    labels[list00[i]] = 0
            random.shuffle(train_data)
            random.shuffle(val_data)
            random.shuffle(test_data)
            train_data0 = []
            val_data0 = []
            test_data0 = []
            for c in range(len(train_data)):
                train_data0.append(tuple(train_data[c]))
            for d in range(len(test_data)):
                test_data0.append(tuple(test_data[d]))
            for e in range(len(val_data)):
                val_data0.append(tuple(val_data[e]))
            for i in range(len(train_data0)):
                if train_data0[i] in labels.keys():
                    train_labels.append(labels[train_data0[i]])
            for j in range(len(test_data0)):
                if test_data0[j] in labels.keys():
                    test_labels.append(labels[test_data0[j]])
            for k in range(len(val_data0)):
                if val_data0[k] in labels.keys():
                    val_labels.append(labels[val_data0[k]])
            edge_index = [[], []]
            for i in range(len(all_data)):
                edge_index[0].append(all_data[i][0])
                edge_index[1].append(all_data[i][1])
            edge_weight = np.ones(len(edge_index[1]))
            edge_weight = torch.FloatTensor(edge_weight)
            edge_index = torch.LongTensor(edge_index)
            query_edges = torch.LongTensor(train_data)
            query_val_edges = torch.LongTensor(val_data)
            y = torch.LongTensor(train_labels)
            y_val = torch.LongTensor(val_labels)
            query_test_edges = torch.LongTensor(test_data)
            y_test = torch.LongTensor(test_labels)
            X = in_out_degree(edge_index, size=len(data.x)).to(device)
            edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(
                edge_index, len(data.x), edge_weight)
            best_auc = 0
            best_aupr = 0
            best_f1 = 0
            best_acc = 0
            for epoch in range(args.epochs):
                train_loss, train_acc, train_auc, train_new_weight = train(X, y, edge_index, edge_in, in_weight,
                                                                           edge_out, out_weight, query_edges)
                edge_weight = torch.FloatTensor(train_new_weight)
                edge_index, edge_in, in_weight, edge_out, out_weight = directed_features_in_out(
                    edge_index, len(data.x), edge_weight)

                test_acc, test_auc, f1, aupr = test(X, y_test, edge_index, edge_in, in_weight, edge_out,
                                                    out_weight, query_test_edges)

                if test_auc > best_auc:
                    best_auc = test_auc
                if aupr > best_aupr:
                    best_aupr = aupr
                if f1 > best_f1:
                    best_f1 = f1
                if test_acc > best_acc:
                    best_acc = test_acc
                all_auc.append(train_auc)
                all_test_auc.append(test_auc)
                print('epoch:{:03d},train_loss:{:.4f},train_acc:{:.4f},train_auc:{:.4f},'
                      'test_acc:{:.4f},test_auc:{:.4f},f1:{:.4f},aupr:{:.4f}'.format
                      (epoch + 1, train_loss, train_acc, train_auc, test_acc, test_auc, f1, aupr))
            locals()[f'best_auc_{net}'].append(best_auc)
            locals()[f'best_aupr_{net}'].append(best_aupr)
            locals()[f'best_f1_{net}'].append(best_f1)
            locals()[f'best_acc_{net}'].append(best_acc)
            print('========================================')
        sum1 += sum2
    print(locals()[f'best_auc_{net}'])
    print(locals()[f'best_aupr_{net}'])
    print(locals()[f'best_f1_{net}'])
    print(locals()[f'best_acc_{net}'])
    print('================================================')
    result.append(sum1 / 25)
auc_c1 = [locals()[f'best_auc_{0}'][0], locals()[f'best_auc_{0}'][1], locals()[f'best_auc_{0}'][2],
          locals()[f'best_auc_{0}'][3], locals()[f'best_auc_{0}'][4]]
auc_c2 = [locals()[f'best_auc_{0}'][5], locals()[f'best_auc_{0}'][6], locals()[f'best_auc_{0}'][7],
          locals()[f'best_auc_{0}'][8], locals()[f'best_auc_{0}'][9]]
auc_c3 = [locals()[f'best_auc_{0}'][10], locals()[f'best_auc_{0}'][11], locals()[f'best_auc_{0}'][12],
          locals()[f'best_auc_{0}'][13], locals()[f'best_auc_{0}'][14]]
auc_c4 = [locals()[f'best_auc_{0}'][15], locals()[f'best_auc_{0}'][16], locals()[f'best_auc_{0}'][17],
          locals()[f'best_auc_{0}'][18], locals()[f'best_auc_{0}'][19]]
auc_c5 = [locals()[f'best_auc_{0}'][20], locals()[f'best_auc_{0}'][21], locals()[f'best_auc_{0}'][22],
          locals()[f'best_auc_{0}'][23], locals()[f'best_auc_{0}'][24]]
auc_h1 = [locals()[f'best_auc_{1}'][0], locals()[f'best_auc_{1}'][1], locals()[f'best_auc_{1}'][2],
          locals()[f'best_auc_{1}'][3], locals()[f'best_auc_{1}'][4]]
auc_h2 = [locals()[f'best_auc_{1}'][5], locals()[f'best_auc_{1}'][6], locals()[f'best_auc_{1}'][7],
          locals()[f'best_auc_{1}'][8], locals()[f'best_auc_{1}'][9]]
auc_h3 = [locals()[f'best_auc_{1}'][10], locals()[f'best_auc_{1}'][11], locals()[f'best_auc_{1}'][12],
          locals()[f'best_auc_{1}'][13], locals()[f'best_auc_{1}'][14]]
auc_h4 = [locals()[f'best_auc_{1}'][15], locals()[f'best_auc_{1}'][16], locals()[f'best_auc_{1}'][17],
          locals()[f'best_auc_{1}'][18], locals()[f'best_auc_{1}'][19]]
auc_h5 = [locals()[f'best_auc_{1}'][20], locals()[f'best_auc_{1}'][21], locals()[f'best_auc_{1}'][22],
          locals()[f'best_auc_{1}'][23], locals()[f'best_auc_{1}'][24]]
auc_o1 = [locals()[f'best_auc_{2}'][0], locals()[f'best_auc_{2}'][1], locals()[f'best_auc_{2}'][2],
          locals()[f'best_auc_{2}'][3], locals()[f'best_auc_{2}'][4]]
auc_o2 = [locals()[f'best_auc_{2}'][5], locals()[f'best_auc_{2}'][6], locals()[f'best_auc_{2}'][7],
          locals()[f'best_auc_{2}'][8], locals()[f'best_auc_{2}'][9]]
auc_o3 = [locals()[f'best_auc_{2}'][10], locals()[f'best_auc_{2}'][11], locals()[f'best_auc_{2}'][12],
          locals()[f'best_auc_{2}'][13], locals()[f'best_auc_{2}'][14]]
auc_o4 = [locals()[f'best_auc_{2}'][15], locals()[f'best_auc_{2}'][16], locals()[f'best_auc_{2}'][17],
          locals()[f'best_auc_{2}'][18], locals()[f'best_auc_{2}'][19]]
auc_o5 = [locals()[f'best_auc_{2}'][20], locals()[f'best_auc_{2}'][21], locals()[f'best_auc_{2}'][22],
          locals()[f'best_auc_{2}'][23], locals()[f'best_auc_{2}'][24]]
print('cold_network的auc均值为：{:.4f},标准差为：{:.4f};'
      'heat_network的auc均值为：{:.4f},标准差为：{:.4f};'
      'oxidative_network的auc均值为：{:.4f},标准差为：{:.4f}'.
      format(np.mean(locals()[f'best_auc_{0}']),
             np.std([np.std(auc_c1, ddof=0), np.std(auc_c2, ddof=0), np.std(auc_c3, ddof=0), np.std(auc_c4, ddof=0),
                     np.std(auc_c5, ddof=0)]),
             np.mean(locals()[f'best_auc_{1}']),
             np.std([np.std(auc_h1, ddof=0), np.std(auc_h2, ddof=0), np.std(auc_h3, ddof=0), np.std(auc_h4, ddof=0),
                     np.std(auc_h5, ddof=0)]),
             np.mean(locals()[f'best_auc_{2}']),
             np.std([np.std(auc_o1, ddof=0), np.std(auc_o2, ddof=0), np.std(auc_o3, ddof=0), np.std(auc_o4, ddof=0),
                     np.std(auc_o5, ddof=0)])))
aupr_c1 = [locals()[f'best_aupr_{0}'][0], locals()[f'best_aupr_{0}'][1], locals()[f'best_aupr_{0}'][2],
           locals()[f'best_aupr_{0}'][3], locals()[f'best_aupr_{0}'][4]]
aupr_c2 = [locals()[f'best_aupr_{0}'][5], locals()[f'best_aupr_{0}'][6], locals()[f'best_aupr_{0}'][7],
           locals()[f'best_aupr_{0}'][8], locals()[f'best_aupr_{0}'][9]]
aupr_c3 = [locals()[f'best_aupr_{0}'][10], locals()[f'best_aupr_{0}'][11], locals()[f'best_aupr_{0}'][12],
           locals()[f'best_aupr_{0}'][13], locals()[f'best_aupr_{0}'][14]]
aupr_c4 = [locals()[f'best_aupr_{0}'][15], locals()[f'best_aupr_{0}'][16], locals()[f'best_aupr_{0}'][17],
           locals()[f'best_aupr_{0}'][18], locals()[f'best_aupr_{0}'][19]]
aupr_c5 = [locals()[f'best_aupr_{0}'][20], locals()[f'best_aupr_{0}'][21], locals()[f'best_aupr_{0}'][22],
           locals()[f'best_aupr_{0}'][23], locals()[f'best_aupr_{0}'][24]]
aupr_h1 = [locals()[f'best_aupr_{1}'][0], locals()[f'best_aupr_{1}'][1], locals()[f'best_aupr_{1}'][2],
           locals()[f'best_aupr_{1}'][3], locals()[f'best_aupr_{1}'][4]]
aupr_h2 = [locals()[f'best_aupr_{1}'][5], locals()[f'best_aupr_{1}'][6], locals()[f'best_aupr_{1}'][7],
           locals()[f'best_aupr_{1}'][8], locals()[f'best_aupr_{1}'][9]]
aupr_h3 = [locals()[f'best_aupr_{1}'][10], locals()[f'best_aupr_{1}'][11], locals()[f'best_aupr_{1}'][12],
           locals()[f'best_aupr_{1}'][13], locals()[f'best_aupr_{1}'][14]]
aupr_h4 = [locals()[f'best_aupr_{1}'][15], locals()[f'best_aupr_{1}'][16], locals()[f'best_aupr_{1}'][17],
           locals()[f'best_aupr_{1}'][18], locals()[f'best_aupr_{1}'][19]]
aupr_h5 = [locals()[f'best_aupr_{1}'][20], locals()[f'best_aupr_{1}'][21], locals()[f'best_aupr_{1}'][22],
           locals()[f'best_aupr_{1}'][23], locals()[f'best_aupr_{1}'][24]]
aupr_o1 = [locals()[f'best_aupr_{2}'][0], locals()[f'best_aupr_{2}'][1], locals()[f'best_aupr_{2}'][2],
           locals()[f'best_aupr_{2}'][3], locals()[f'best_aupr_{2}'][4]]
aupr_o2 = [locals()[f'best_aupr_{2}'][5], locals()[f'best_aupr_{2}'][6], locals()[f'best_aupr_{2}'][7],
           locals()[f'best_aupr_{2}'][8], locals()[f'best_aupr_{2}'][9]]
aupr_o3 = [locals()[f'best_aupr_{2}'][10], locals()[f'best_aupr_{2}'][11], locals()[f'best_aupr_{2}'][12],
           locals()[f'best_aupr_{2}'][13], locals()[f'best_aupr_{2}'][14]]
aupr_o4 = [locals()[f'best_aupr_{2}'][15], locals()[f'best_aupr_{2}'][16], locals()[f'best_aupr_{2}'][17],
           locals()[f'best_aupr_{2}'][18], locals()[f'best_aupr_{2}'][19]]
aupr_o5 = [locals()[f'best_aupr_{2}'][20], locals()[f'best_aupr_{2}'][21], locals()[f'best_aupr_{2}'][22],
           locals()[f'best_aupr_{2}'][23], locals()[f'best_aupr_{2}'][24]]
print('cold_network的aupr均值为：{:.4f},标准差为：{:.4f};'
      'heat_network的aupr均值为：{:.4f},标准差为：{:.4f};'
      'oxidative_network的aupr均值为：{:.4f},标准差为：{:.4f}'.
      format(np.mean(locals()[f'best_aupr_{0}']),
             np.std([np.std(aupr_c1, ddof=0), np.std(aupr_c2, ddof=0), np.std(aupr_c3, ddof=0), np.std(aupr_c4, ddof=0),
                     np.std(aupr_c5, ddof=0)]),
             np.mean(locals()[f'best_aupr_{1}']),
             np.std([np.std(aupr_h1, ddof=0), np.std(aupr_h2, ddof=0), np.std(aupr_h3, ddof=0), np.std(aupr_h4, ddof=0),
                     np.std(aupr_h5, ddof=0)]),
             np.mean(locals()[f'best_aupr_{2}']),
             np.std([np.std(aupr_o1, ddof=0), np.std(aupr_o2, ddof=0), np.std(aupr_o3, ddof=0), np.std(aupr_o4, ddof=0),
                     np.std(aupr_o5, ddof=0)])))
f1_c1 = [locals()[f'best_f1_{0}'][0], locals()[f'best_f1_{0}'][1], locals()[f'best_f1_{0}'][2],
         locals()[f'best_f1_{0}'][3], locals()[f'best_f1_{0}'][4]]
f1_c2 = [locals()[f'best_f1_{0}'][5], locals()[f'best_f1_{0}'][6], locals()[f'best_f1_{0}'][7],
         locals()[f'best_f1_{0}'][8], locals()[f'best_f1_{0}'][9]]
f1_c3 = [locals()[f'best_f1_{0}'][10], locals()[f'best_f1_{0}'][11], locals()[f'best_f1_{0}'][12],
         locals()[f'best_f1_{0}'][13], locals()[f'best_f1_{0}'][14]]
f1_c4 = [locals()[f'best_f1_{0}'][15], locals()[f'best_f1_{0}'][16], locals()[f'best_f1_{0}'][17],
         locals()[f'best_f1_{0}'][18], locals()[f'best_f1_{0}'][19]]
f1_c5 = [locals()[f'best_f1_{0}'][20], locals()[f'best_f1_{0}'][21], locals()[f'best_f1_{0}'][22],
         locals()[f'best_f1_{0}'][23], locals()[f'best_f1_{0}'][24]]
f1_h1 = [locals()[f'best_f1_{1}'][0], locals()[f'best_f1_{1}'][1], locals()[f'best_f1_{1}'][2],
         locals()[f'best_f1_{1}'][3], locals()[f'best_f1_{1}'][4]]
f1_h2 = [locals()[f'best_f1_{1}'][5], locals()[f'best_f1_{1}'][6], locals()[f'best_f1_{1}'][7],
         locals()[f'best_f1_{1}'][8], locals()[f'best_f1_{1}'][9]]
f1_h3 = [locals()[f'best_f1_{1}'][10], locals()[f'best_f1_{1}'][11], locals()[f'best_f1_{1}'][12],
         locals()[f'best_f1_{1}'][13], locals()[f'best_f1_{1}'][14]]
f1_h4 = [locals()[f'best_f1_{1}'][15], locals()[f'best_f1_{1}'][16], locals()[f'best_f1_{1}'][17],
         locals()[f'best_f1_{1}'][18], locals()[f'best_f1_{1}'][19]]
f1_h5 = [locals()[f'best_f1_{1}'][20], locals()[f'best_f1_{1}'][21], locals()[f'best_f1_{1}'][22],
         locals()[f'best_f1_{1}'][23], locals()[f'best_f1_{1}'][24]]
f1_o1 = [locals()[f'best_f1_{2}'][0], locals()[f'best_f1_{2}'][1], locals()[f'best_f1_{2}'][2],
         locals()[f'best_f1_{2}'][3], locals()[f'best_f1_{2}'][4]]
f1_o2 = [locals()[f'best_f1_{2}'][5], locals()[f'best_f1_{2}'][6], locals()[f'best_f1_{2}'][7],
         locals()[f'best_f1_{2}'][8], locals()[f'best_f1_{2}'][9]]
f1_o3 = [locals()[f'best_f1_{2}'][10], locals()[f'best_f1_{2}'][11], locals()[f'best_f1_{2}'][12],
         locals()[f'best_f1_{2}'][13], locals()[f'best_f1_{2}'][14]]
f1_o4 = [locals()[f'best_f1_{2}'][15], locals()[f'best_f1_{2}'][16], locals()[f'best_f1_{2}'][17],
         locals()[f'best_f1_{2}'][18], locals()[f'best_f1_{2}'][19]]
f1_o5 = [locals()[f'best_f1_{2}'][20], locals()[f'best_f1_{2}'][21], locals()[f'best_f1_{2}'][22],
         locals()[f'best_f1_{2}'][23], locals()[f'best_f1_{2}'][24]]
print('cold_network的f1均值为：{:.4f},标准差为：{:.4f};'
      'heat_network的f1均值为：{:.4f},标准差为：{:.4f};'
      'oxidative_network的f1均值为：{:.4f},标准差为：{:.4f}'.
      format(np.mean(locals()[f'best_f1_{0}']),
             np.std([np.std(f1_c1, ddof=0), np.std(f1_c2, ddof=0), np.std(f1_c3, ddof=0), np.std(f1_c4, ddof=0),
                     np.std(f1_c5, ddof=0)]),
             np.mean(locals()[f'best_f1_{1}']),
             np.std([np.std(f1_h1, ddof=0), np.std(f1_h2, ddof=0), np.std(f1_h3, ddof=0), np.std(f1_h4, ddof=0),
                     np.std(f1_h5, ddof=0)]),
             np.mean(locals()[f'best_f1_{2}']),
             np.std([np.std(f1_o1, ddof=0), np.std(f1_o2, ddof=0), np.std(f1_o3, ddof=0), np.std(f1_o4, ddof=0),
                     np.std(f1_o5, ddof=0)])))
acc_c1 = [locals()[f'best_acc_{0}'][0], locals()[f'best_acc_{0}'][1], locals()[f'best_acc_{0}'][2],
          locals()[f'best_acc_{0}'][3], locals()[f'best_acc_{0}'][4]]
acc_c2 = [locals()[f'best_acc_{0}'][5], locals()[f'best_acc_{0}'][6], locals()[f'best_acc_{0}'][7],
          locals()[f'best_acc_{0}'][8], locals()[f'best_acc_{0}'][9]]
acc_c3 = [locals()[f'best_acc_{0}'][10], locals()[f'best_acc_{0}'][11], locals()[f'best_acc_{0}'][12],
          locals()[f'best_acc_{0}'][13], locals()[f'best_acc_{0}'][14]]
acc_c4 = [locals()[f'best_acc_{0}'][15], locals()[f'best_acc_{0}'][16], locals()[f'best_acc_{0}'][17],
          locals()[f'best_acc_{0}'][18], locals()[f'best_acc_{0}'][19]]
acc_c5 = [locals()[f'best_acc_{0}'][20], locals()[f'best_acc_{0}'][21], locals()[f'best_acc_{0}'][22],
          locals()[f'best_acc_{0}'][23], locals()[f'best_acc_{0}'][24]]
acc_h1 = [locals()[f'best_acc_{1}'][0], locals()[f'best_acc_{1}'][1], locals()[f'best_acc_{1}'][2],
          locals()[f'best_acc_{1}'][3], locals()[f'best_acc_{1}'][4]]
acc_h2 = [locals()[f'best_acc_{1}'][5], locals()[f'best_acc_{1}'][6], locals()[f'best_acc_{1}'][7],
          locals()[f'best_acc_{1}'][8], locals()[f'best_acc_{1}'][9]]
acc_h3 = [locals()[f'best_acc_{1}'][10], locals()[f'best_acc_{1}'][11], locals()[f'best_acc_{1}'][12],
          locals()[f'best_acc_{1}'][13], locals()[f'best_acc_{1}'][14]]
acc_h4 = [locals()[f'best_acc_{1}'][15], locals()[f'best_acc_{1}'][16], locals()[f'best_acc_{1}'][17],
          locals()[f'best_acc_{1}'][18], locals()[f'best_acc_{1}'][19]]
acc_h5 = [locals()[f'best_acc_{1}'][20], locals()[f'best_acc_{1}'][21], locals()[f'best_acc_{1}'][22],
          locals()[f'best_acc_{1}'][23], locals()[f'best_acc_{1}'][24]]
acc_o1 = [locals()[f'best_acc_{2}'][0], locals()[f'best_acc_{2}'][1], locals()[f'best_acc_{2}'][2],
          locals()[f'best_acc_{2}'][3], locals()[f'best_acc_{2}'][4]]
acc_o2 = [locals()[f'best_acc_{2}'][5], locals()[f'best_acc_{2}'][6], locals()[f'best_acc_{2}'][7],
          locals()[f'best_acc_{2}'][8], locals()[f'best_acc_{2}'][9]]
acc_o3 = [locals()[f'best_acc_{2}'][10], locals()[f'best_acc_{2}'][11], locals()[f'best_acc_{2}'][12],
          locals()[f'best_acc_{2}'][13], locals()[f'best_acc_{2}'][14]]
acc_o4 = [locals()[f'best_acc_{2}'][15], locals()[f'best_acc_{2}'][16], locals()[f'best_acc_{2}'][17],
          locals()[f'best_acc_{2}'][18], locals()[f'best_acc_{2}'][19]]
acc_o5 = [locals()[f'best_acc_{2}'][20], locals()[f'best_acc_{2}'][21], locals()[f'best_acc_{2}'][22],
          locals()[f'best_acc_{2}'][23], locals()[f'best_acc_{2}'][24]]
print('cold_network的acc均值为：{:.4f},标准差为：{:.4f};'
      'heat_network的acc均值为：{:.4f},标准差为：{:.4f};'
      'oxidative_network的acc均值为：{:.4f},标准差为：{:.4f}'.
      format(np.mean(locals()[f'best_acc_{0}']),
             np.std([np.std(acc_c1, ddof=0), np.std(acc_c2, ddof=0), np.std(acc_c3, ddof=0), np.std(acc_c4, ddof=0),
                     np.std(acc_c5, ddof=0)]),
             np.mean(locals()[f'best_acc_{1}']),
             np.std([np.std(acc_h1, ddof=0), np.std(acc_h2, ddof=0), np.std(acc_h3, ddof=0), np.std(acc_h4, ddof=0),
                     np.std(acc_h5, ddof=0)]),
             np.mean(locals()[f'best_acc_{2}']),
             np.std([np.std(acc_o1, ddof=0), np.std(acc_o2, ddof=0), np.std(acc_o3, ddof=0), np.std(acc_o4, ddof=0),
                     np.std(acc_o5, ddof=0)])))
