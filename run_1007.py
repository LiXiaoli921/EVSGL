
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os

# 加载基因表达数据
def load_gene_expression_data(path):
    gene_expression_df = pd.read_csv(path, index_col=0)
    return gene_expression_df

# 提取MRI影像特征并与基因数据配对
def extract_mri_features(image_dir, sample_ids):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()  # 去除最后一层
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])

    features = []
    valid_sample_ids = []
    for sample_id in sample_ids:
        sample_dir = os.path.join(image_dir, sample_id)  # 假设每个 sample_id 对应一个文件夹
        if not os.path.exists(sample_dir):
            print(f'文件夹 {sample_dir} 未找到，跳过该样本。')
            continue

        # 读取文件夹中的所有图像文件
        image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))]
        if not image_files:
            print(f'文件夹 {sample_dir} 中没有找到图像文件，跳过该样本。')
            continue

        for image_file in image_files:
            img_path = os.path.join(sample_dir, image_file)
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                out = model(batch_t)
            features.append(out.numpy())

        valid_sample_ids.append(sample_id)

    features = np.vstack(features)
    return features, valid_sample_ids

# 构建相似性图
def build_similarity_graph(features, k=5):
    similarity_matrix = cosine_similarity(features)
    adjacency_matrix = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        idx = np.argsort(-similarity_matrix[i, :])[:k+1]
        adjacency_matrix[i, idx] = similarity_matrix[i, idx]
    # 构建边列表
    edge_index = np.array(np.nonzero(adjacency_matrix))
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_weight, dtype=torch.float)

# 定义视图编码器
class ViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ViewEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# 定义融合层
class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_views):
        super(FusionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(num_views))
        nn.init.uniform_(self.attention_weights)

    def forward(self, view_reps):
        weights = torch.softmax(self.attention_weights, dim=0)
        fused_rep = 0
        for i, rep in enumerate(view_reps):
            fused_rep += weights[i] * rep
        return fused_rep

# 定义EVSGL模型
class EVSGL_Model(nn.Module):
    def __init__(self, input_dims, hidden_dim, num_clusters):
        super(EVSGL_Model, self).__init__()
        self.view_encoders = nn.ModuleList([
            ViewEncoder(input_dim, hidden_dim) for input_dim in input_dims
        ])
        self.fusion_layer = FusionLayer(hidden_dim, len(input_dims))
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, x_list, edge_index_list, edge_weight_list):
        view_reps = []
        for i in range(len(self.view_encoders)):
            view_rep = self.view_encoders[i](x_list[i],edge_index_list[i],edge_weight_list[i])
            view_reps.append(view_rep)
        fused_rep = self.fusion_layer(view_reps)
        q = self.soft_assignment(fused_rep)
        return fused_rep, q

    def soft_assignment(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, 2))
        q = q.pow((1.0 + 1.0)/2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

# 目标分布计算
def target_distribution(q):
    weight = (q ** 2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# 训练模型
def train_evsgl(model, x_list, edge_index_list, edge_weight_list, num_clusters, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        fused_rep, q = model(x_list, edge_index_list, edge_weight_list)
        p = target_distribution(q.detach())
        kl_loss = nn.KLDivLoss(reduction='batchmean')(torch.log(q), p)
        loss = kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    return fused_rep

# 主函数
def main():
    # 数据路径
    # gene_expression_path = 'path_to_gene_expression.csv'  # 替换为实际的基因数据文件路径
    # mri_image_dir = 'path_to_mri_images/'                 # 替换为实际的MRI图像目录路径
    gene_expression_path = 'data/gene/breast_expr_c90.csv'
    # gene_expression_path = 'data/gene/breast_expr_all.csv'
    # gene_expression_path = '/home/lior/PycharmProjects/EVSGL/data/gene/breast_expr_all.csv'


    mri_image_dir = 'data/BRCA_select/'
    # mri_image_dir = 'data/BRCA_jpg/'

    # 加载基因数据
    # gene_expression_df = load_gene_expression_data(gene_expression_path)
    gene_expression_df = pd.read_csv(gene_expression_path, index_col=0)

    # 假设CSV文件第一行为样本ID，第一列为基因名，其余列为基因表达值

    # 获取样本ID列表（从第二列开始，因为第一列是基因名）
    sample_ids = gene_expression_df.columns.tolist()

    # 转置数据，使得行是样本，列是基因
    gene_expression_df = gene_expression_df.transpose()

    # 将基因表达值转换为numpy数组
    # 注意，此时gene_expression_df的行索引为样本ID，列名为基因名
    gene_expression = gene_expression_df.values

    # 提取与基因数据配对的MRI影像特征
    image_features, valid_sample_ids = extract_mri_features(mri_image_dir, sample_ids)

    # 根据有效的样本ID过滤基因数据
    gene_expression_df = gene_expression_df.loc[valid_sample_ids]
    gene_expression = gene_expression_df.values

    # 构建相似性图
    edge_index_gene, edge_weight_gene = build_similarity_graph(gene_expression)
    edge_index_image, edge_weight_image = build_similarity_graph(image_features)

    # 转换为张量
    x_gene = torch.tensor(gene_expression, dtype=torch.float)
    x_image = torch.tensor(image_features, dtype=torch.float)

    # 准备输入列表
    x_list = [x_gene, x_image]
    edge_index_list = [edge_index_gene, edge_index_image]
    edge_weight_list = [edge_weight_gene, edge_weight_image]
    input_dims = [x.shape[1] for x in x_list]

    # 定义模型
    hidden_dim = 128
    num_clusters = 5  # 根据数据调整聚类数目
    model = EVSGL_Model(input_dims, hidden_dim, num_clusters)

    # 训练模型
    fused_rep = train_evsgl(model, x_list, edge_index_list, edge_weight_list, num_clusters)

    # 聚类结果
    fused_rep_numpy = fused_rep.detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    y_pred = kmeans.fit_predict(fused_rep_numpy)

    # 输出聚类结果
    print("聚类结果:", y_pred)

if __name__ == '__main__':
    main()