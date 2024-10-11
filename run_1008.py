import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from layers import *
import numpy as np
from graph import *
from utils import *

class SharedEncoder(nn.ModuleList):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2):
        super(SharedEncoder, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.input_feat_dim = input_feat_dim
        self.encoder = nn.ModuleList([
            GraphConvolution(input_feat_dim, hidden_dim1, act=F.relu),
            GraphConvolution(hidden_dim1, hidden_dim2, act=lambda x:x)
        ])

    def forward(self,fea, adj):
        hidden = self.encoder[0](fea,adj)
        embedding = self.encoder[1](hidden,adj)
        return embedding

class GMIM(nn.ModuleList):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GMIM, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.input_feat_dim = input_feat_dim
        self.act = nn.PReLU()
        self.disc1 = Discriminator(input_feat_dim, hidden_dim2)
        self.disc2 = Discriminator(hidden_dim2,input_feat_dim)

        self.VSEncoder = nn.ModuleList([
            GraphConvolution(input_feat_dim, hidden_dim1, act=F.relu),
            GraphConvolution(hidden_dim1, hidden_dim2, act=lambda x:x)
        ])
        self.decoder = InnerProductDecoderWithoutW(dropout)

    def forward(self,fea, adj, raw_adj, neg_num=5):
        hidden = self.VSEncoder[0](fea,adj)
        embedding = self.VSEncoder[1](hidden,adj)

        # node feature mutual information
        h_neighbor = self.act(torch.unsqueeze(torch.spmm(raw_adj, fea), 0))
        res_mi_pos, res_mi_neg = self.disc1(embedding, fea, negative_sampling(raw_adj, neg_num))
        res_local_pos, res_local_neg = self.disc2(torch.squeeze(h_neighbor,0), embedding, negative_sampling(raw_adj, neg_num))

        # Topological mutual information
        recons_graph = self.decoder(embedding)

        return embedding, recons_graph, res_mi_pos, res_mi_neg, res_local_pos, res_local_neg

class EVSGL(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, hidden_dim3,hidden_dim4, dropout, cluster_num, numView, data_type):
        super(EVSGL, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4
        self.numView = numView
        self.cluster_num = cluster_num
        self.data_type = data_type

        self.input_feat_dim = input_feat_dim

        if self.data_type == "text":
            consistent_input_dim = self.hidden_dim2*self.numView
        else:
            consistent_input_dim = self.hidden_dim2

        self.ConsistentEncoder = nn.ModuleList([
            GraphConvolution(consistent_input_dim, hidden_dim3, act=F.relu),
            GraphConvolution(hidden_dim3, hidden_dim4, act=lambda x:x)
        ])

        self.W_h = self.init_uniform_weight(self.numView)
        self.W_g = self.init_uniform_weight(self.numView)

        self.SharedEncoders = nn.ModuleList()
        for i in range(self.numView):
            self.SharedEncoders.append(SharedEncoder(input_feat_dim[i], hidden_dim1, hidden_dim2).cuda())

        self.cluster_layer = ClusterLayer(self.cluster_num, hidden_dim4)

        self.ConsistentDecoders = nn.ModuleList()
        for i in range(self.numView):
            self.ConsistentDecoders.append(InnerProductDecoder(dropout, self.hidden_dim4))

    def init_uniform_weight(self, viewNum):
        return nn.Parameter(torch.ones(viewNum)/viewNum, requires_grad=True)

    def View_Wise_Attention_Fusion(self, embs, adjs):

        # sum up probabilitic graphs
        norm_theta_graph = 0
        for i in range(self.numView):
            norm_theta_graph += torch.exp(self.W_g[i])

        fusedA = 0
        for i in range(self.numView):
            fusedA = fusedA + (torch.exp(self.W_g[i]) / norm_theta_graph) * adjs[i]

        # combine embedding with concat or sum
        norm_theta_fea = 0
        for i in range(self.numView):
            norm_theta_fea += torch.exp(self.W_h[i])

        if self.data_type == "text": #bbcsport
            fusedH = []
            for i in range(self.numView):
                fusedH.append(embs[i] * (torch.exp(self.W_h[i]) / norm_theta_fea))
            combinedH = torch.cat(fusedH, 1)
        else:
            #others(images)
            combinedH = 0
            for i in range(self.numView):
                combinedH = combinedH + (torch.exp(self.W_h[i]) / norm_theta_fea) * embs[i]

        fusion_hidden = self.ConsistentEncoder[0](combinedH,fusedA)
        fusion_embedding = self.ConsistentEncoder[1](fusion_hidden,fusedA)
        return fusion_embedding

    def process_graph(self, fea, num_neighbors, graph_type="euclidean"):
        adj, raw_adj = probability_graph(fea.t(), num_neighbors, graph_type)  # first
        adj = adj.detach()
        raw_adj = raw_adj.detach()
        Laplacian = get_Laplacian_from_adj(adj)
        return adj, Laplacian, raw_adj

    def forward(self, feats, adjs):
        specific_embeddings = []
        for i in range(self.numView):
            specific_embedding = self.SharedEncoders[i](feats[i],adjs[i])
            specific_embeddings.append(specific_embedding)

        fusion_embedding = self.View_Wise_Attention_Fusion(specific_embeddings, adjs)
        recons_graphs = []
        for i in range(self.numView):
            recons_graphs.append(self.ConsistentDecoders[i](fusion_embedding))

        q = self.cluster_layer(fusion_embedding)
        return fusion_embedding, recons_graphs, q

    def visualize_clusters(self, embedding, labels):
        # 使用 t-SNE 将嵌入降维到 2D
        tsne = TSNE(n_components=2, random_state=42)
        embedding_2d = tsne.fit_transform(embedding.detach().cpu().numpy())

        # 绘制聚类结果
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title("t-SNE Visualization of Clustered Nodes")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.show()

class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, hidden_dim,  act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.reset_parameters()

    def get_init_weight(self, shape):
        bound = np.sqrt(6.0 / (np.sum(shape)))
        ini = torch.rand(shape) * 2 * bound - bound
        return torch.nn.Parameter(ini, requires_grad=True)

    def reset_parameters(self):
        self.weight= self.get_init_weight([self.hidden_dim, self.hidden_dim])

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        tmp = torch.mm(z, self.weight)
        adj = self.act(torch.mm(tmp, z.t()))
        return adj

class InnerProductDecoderWithoutW(nn.Module):

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoderWithoutW, self).__init__()

        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj

# 模型训练和可视化示例

if __name__ == "__main__":
    # 假设已经准备好了特征和邻接矩阵
    # feats = [gene_features_tensor, image_features_tensor]
    # adjs = [gene_adj_matrix, image_adj_matrix]
    # labels = ground_truth_labels  # 如果有真实的标签

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_samples = feats[0].shape[0]
    gene_feature_dim = feats[0].shape[1]
    image_feature_dim = feats[1].shape[1]

    # 定义模型
    model = EVSGL(
        input_feat_dim=[gene_feature_dim, image_feature_dim],
        hidden_dim1=256,
        hidden_dim2=128,
        hidden_dim3=64,
        hidden_dim4=32,
        dropout=0.5,
        cluster_num=3,  # 根据需要设置聚类数目
        numView=2,
        data_type='custom'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        fusion_embedding, recons_graphs, q = model(feats, adjs)
        # 计算损失函数（重构损失、聚类损失等）
        loss = compute_loss(...)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # 获取聚类结果
    with torch.no_grad():
        model.eval()
        fusion_embedding, _, q = model(feats, adjs)
        # 提取聚类标签
        cluster_labels = torch.argmax(q, dim=1).cpu().numpy()

    # 可视化聚类结果
    model.visualize_clusters(fusion_embedding, cluster_labels)