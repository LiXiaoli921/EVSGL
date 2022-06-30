import numpy as np
import scipy.io as sio
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def sp_func(arg):
    return torch.log(1+torch.exp(arg))

def mi_loss_jsd(pos, neg):
    e_pos = torch.mean(sp_func(-pos))
    e_neg = torch.mean(torch.mean(sp_func(neg),0))
    return e_pos+e_neg

def minMaxNormalize(x, min=0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:  # min=-1
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x

def negative_sampling(adj_ori, sample_times):
    sample_list = []
    for j in range(sample_times):
        sample_iter = []
        i = 0
        while True:
            randnum = np.random.randint(0,adj_ori.shape[0])
            if randnum!=i:
                sample_iter.append(randnum)
                i = i+1
            if len(sample_iter)==adj_ori.shape[0]:
                break
        sample_list.append(sample_iter)
    return sample_list


# The preprocess for all datasets will be updated after the paper is accepted

def load_data(dataset):
    if dataset =="bbcsport":
        dataset_path = "./data/BBCSport.mat"


    data = sio.loadmat(dataset_path)
    print(dataset)

    if dataset == "bbcsport":
        X = data["fea"][0]
        y_true = np.squeeze(data['gt'])
        for v in range(2):
            X[v] = normalize(X[v].toarray())
        numView = 2
        X = np.array(X) / 1.0
    
    return X, numView, y_true


def plot_embedding(data, label, title):
    fig = plt.figure(figsize=(8, 8))
    tsne_df = pd.DataFrame({'X': data[:, 0],
                            'Y': data[:, 1],
                            'point': label})

    num_classes = len(np.unique(label))
    palette = list(sns.color_palette("hls", num_classes))
    sns.scatterplot(x="X", y="Y",
                    hue='point',
                    palette=palette,
                    legend=False,
                    data=tsne_df)
    return fig
