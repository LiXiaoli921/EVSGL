from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
from sklearn.manifold import TSNE
import argparse
import time
from sklearn.cluster import KMeans

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim

from model import *
import torch.nn.functional as F
from utils import *
from metrics import *
from graph import *
from torch.optim import RMSprop
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--gmim_epochs', type=int, default=10000, help='Number of GMIM epoch.')
parser.add_argument('--fusion_epochs', type=int, default=10000, help='number of fusion epoch')
parser.add_argument('--finetune_epochs', type=int, default=300, help='number of fine-tune epoch')
parser.add_argument('--hidden1', type=int, default=512, help='Number of units in view-specific encoder layer 1.')
parser.add_argument('--hidden2', type=int, default=256, help='Number of units in view-specific encoder layer 2.')
parser.add_argument('--hidden3', type=int, default=128, help='Number of units in consistent encoder layer 1.')
parser.add_argument('--hidden4', type=int, default=32, help='Number of units in consistent encoder layer 2.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='bbcsport', help='type of dataset.')
parser.add_argument('--num_neighbors', type=int, default=60, help='Initialize the number of k-neighbors')
parser.add_argument('--lambda1', type=float, default=1, help='parameter of lambda1')
parser.add_argument('--lambda2', type=float, default=1, help='parameter of lambda2')
parser.add_argument('--mvc_early_stop', type=int, default=200, help='early stop of gmim and fusion')
parser.add_argument('--ft_early_stop', type=int, default=100, help='early stop of fine-tune')


args = parser.parse_args()

# The preprocess for all datasets will be updated after the paper is accepted
cluster_nums = {
    'bbcsport':5,
}

cluster_num = cluster_nums[args.dataset_str]
mvc_early_stop = args.mvc_early_stop
ft_early_stop = args.ft_early_stop
lambda1 = args.lambda1
lambda2 = args.lambda2


def Run_EVSGL(args):
    print("Using {} dataset".format(args.dataset_str))
    features, numView, labels = load_data(args.dataset_str)
    features_dims = [fea.shape[1] for fea in features]
    inputs = []
    for v in range(numView):
        inputs.append(torch.FloatTensor(features[v]).cuda())
    num_neighbors = args.num_neighbors
    torch.cuda.empty_cache()

    data_type = "image"
    if args.dataset_str == "bbcsport" or args.dataset_str == "CiteSeer":
        data_type = "text"
    else:
        data_type = "image"

    print(data_type)
    model = EVSGL(features_dims, args.hidden1, args.hidden2, args.hidden3, args.hidden4, args.dropout, cluster_num, numView, data_type).to(device)

    Raw_adj_list = []
    Laplacian_list = []

    if data_type == "text":
        graph_type="cosine"
    else:
        graph_type="euclidean"

    for v in range(numView):
        torch.cuda.empty_cache()
        _, Laplacian, raw_adj = model.process_graph(inputs[v], num_neighbors, graph_type=graph_type)
        Laplacian_list.append(Laplacian)
        Raw_adj_list.append(raw_adj)

    for i in range(numView):
        raw_adj = Raw_adj_list[i]
        # raw_adj_csc = sp.csr_matrix(raw_adj.detach().cpu(), dtype=np.float32)
        gmim = GMIM(features_dims[i], args.hidden1, args.hidden2, args.dropout).cuda()
        optimizer = optim.Adam(gmim.parameters(), lr=args.lr, weight_decay=1e-5)
        patience = 0
        best_dict = 0
        adj_hat = Laplacian_list[i].cuda()
        norm = adj_hat.shape[0] * adj_hat.shape[0] / float((adj_hat.shape[0] * adj_hat.shape[0] - adj_hat.sum()) * 2)
        pos_weight = float(adj_hat.shape[0] * adj_hat.shape[0] - adj_hat.sum()) / adj_hat.sum()
        best_loss = float("inf")
        for epoch in range(args.gmim_epochs):
            gmim.train()
            fea = inputs[i].cuda()
            optimizer.zero_grad()
            embedding, recons_graph, res_mi_pos, res_mi_neg, res_local_pos, res_local_neg = gmim(fea, adj_hat, raw_adj, 5)

            # Node feature mutual information maximization
            FMIM_Loss = mi_loss_jsd(res_mi_pos, res_mi_neg) + mi_loss_jsd(res_local_pos, res_local_neg)

            # Topological mutual information maximization
            TMIM_Loss = norm * F.binary_cross_entropy_with_logits(recons_graph.view(-1), raw_adj.view(-1), pos_weight=pos_weight)

            total_loss = FMIM_Loss + lambda1*TMIM_Loss

            total_loss.backward()
            optimizer.step(closure=None)
            loss_value = float(total_loss.item())
            if loss_value < best_loss:
                best_loss = loss_value
                best_dict = gmim.VSEncoder.state_dict()
                patience = 0
            else:
                patience += 1
            if patience > mvc_early_stop:
                break

        model_dict = model.SharedEncoders[i].encoder.state_dict()
        model_dict.update(best_dict)
        model.SharedEncoders[i].encoder.load_state_dict(model_dict)


    # View-wise graph fusion
    fusion_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_loss = float('inf')
    patience = 0
    for epoch in range(args.fusion_epochs):
        model.train()
        fusion_optimizer.zero_grad()
        fusion_embedding, recons_graphs, q = model(inputs, Laplacian_list)
        fusion_loss = 0
        for i in range(numView):
            norm = Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] / float((Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] - Laplacian_list[i].sum()) * 2)
            pos_weight = float(Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] - Laplacian_list[i].sum()) / Laplacian_list[i].sum()
            fusion_loss += norm * F.binary_cross_entropy_with_logits(recons_graphs[i].view(-1), Raw_adj_list[i].view(-1), pos_weight=pos_weight)

        fusion_loss.backward(retain_graph=True)
        fusion_optimizer.step(closure=None)
        loss_value = float(fusion_loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            patience = 0
        else:
            patience += 1

        if patience > mvc_early_stop:
            break

    model.eval()
    fusion_embedding, _, q = model(inputs, Laplacian_list)
    p = target_distribution(q)

    kmeans = KMeans(n_clusters=cluster_num).fit(fusion_embedding.detach().cpu().numpy())
    predict_labels = kmeans.predict(fusion_embedding.detach().cpu().numpy())


    # clustering-friendly fine-tuning
    model.cluster_layer.network.data = torch.tensor(kmeans.cluster_centers_).float().cuda()

    best_loss = float('inf')
    patience = 0
    for epoch in range(args.finetune_epochs):
        model.train()
        # update p with interval T=30
        if (epoch + 1) % 30 == 0:
            _, _, tmp_q = model(inputs, Laplacian_list)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

        fusion_optimizer.zero_grad()
        fusion_embedding, recons_graphs, q = model(inputs, Laplacian_list)
        recons_loss = 0
        for i in range(numView):
            norm = Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] / float((Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] - Laplacian_list[i].sum()) * 2)
            pos_weight = float(Laplacian_list[i].shape[0] * Laplacian_list[i].shape[0] - Laplacian_list[i].sum()) / Laplacian_list[i].sum()

            recons_loss += (norm * F.binary_cross_entropy_with_logits(recons_graphs[i].view(-1), Raw_adj_list[i].view(-1), pos_weight=pos_weight))

        loss_kl = 1.0 * F.kl_div(q.log(), p)

        finetune_loss =  recons_loss + lambda2 * loss_kl
        finetune_loss.backward(retain_graph=True)
        fusion_optimizer.step(closure=None)
        loss_value = float(finetune_loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            patience = 0
        else:
            patience += 1
        if patience > ft_early_stop:
            break

    model.eval()
    fusion_embedding, _, q = model(inputs, Laplacian_list)

    kmeans = KMeans(n_clusters=cluster_num).fit(fusion_embedding.detach().cpu().numpy())
    predict_labels = kmeans.predict(fusion_embedding.detach().cpu().numpy())
    cm = clustering_metrics(labels, predict_labels)
    acc, f1_macro, precision_macro, nmi, adjscore = cm.evaluationClusterModelFromLabel()
    print(
        'ACC=%f, F1=%f, precision_macro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (
        acc, f1_macro, precision_macro, nmi, adjscore))

    return acc, f1_macro, precision_macro, nmi, adjscore


if __name__ == '__main__':
    times = 10

    accs = []
    f1_macros = []
    precision_macros = []
    nmis = []
    adjscores = []
    for t in range(times):
        bg = time.time()
        acc, f1_macro, precision_macro, nmi, adjscore = Run_EVSGL(args)
        ed = time.time()
        accs.append(acc)
        f1_macros.append(f1_macro)
        precision_macros.append(precision_macro)
        nmis.append(nmi)
        adjscores.append(adjscore)

    acc_mean = np.mean(np.array(accs))
    acc_std = np.std(np.array(accs), ddof=1)
    f1_mean = np.mean(np.array(f1_macros))
    f1_std = np.std(np.array(f1_macros), ddof=1)
    precision_mean = np.mean(np.array(precision_macros))
    precision_std = np.std(np.array(precision_macros), ddof=1)
    nmi_mean = np.mean(np.array(nmis))
    nmi_std = np.std(np.array(nmis), ddof=1)
    ari_mean = np.mean(np.array(adjscores))
    ari_std = np.std(np.array(adjscores), ddof=1)
    print(
        'ACC_mean=%f, ACC_std=%f, f1_mean=%f, f1_std=%f, precision_mean=%f, precision_std=%f, nmi_mean=%f, nmi_std=%f, ari_mean=%f, ari_std=%f' % (
            acc_mean, acc_std, f1_mean, f1_std, precision_mean, precision_std, nmi_mean, nmi_std, ari_mean, ari_std))
    print(args)
