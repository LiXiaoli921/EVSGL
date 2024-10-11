import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans

# Extract MRI features and aggregate per sample
def extract_mri_features(image_dir, sample_ids):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()  # Remove the final layer
    model.eval()

    preprocess = weights.transforms()

    features = []
    valid_sample_ids = []
    for sample_id in sample_ids:
        sample_dir = os.path.join(image_dir, sample_id)  # Each sample_id corresponds to a folder
        if not os.path.exists(sample_dir):
            print(f'Folder {sample_dir} not found, skipping this sample.')
            continue

        # Read all image files in the folder
        image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f'No image files found in {sample_dir}, skipping this sample.')
            continue

        sample_features = []
        for image_file in image_files:
            img_path = os.path.join(sample_dir, image_file)
            img = Image.open(img_path).convert('RGB')
            img_t = preprocess(img)
            batch_t = torch.unsqueeze(img_t, 0)

            with torch.no_grad():
                out = model(batch_t)
            sample_features.append(out.numpy())

        if sample_features:
            # Average features for this sample
            sample_features = np.mean(sample_features, axis=0)
            features.append(sample_features)
            valid_sample_ids.append(sample_id)

    features = np.vstack(features)
    return features, valid_sample_ids

# Build similarity graph
def build_similarity_graph(features, k=5):
    similarity_matrix = cosine_similarity(features)
    adjacency_matrix = np.zeros_like(similarity_matrix)
    for i in range(similarity_matrix.shape[0]):
        idx = np.argsort(-similarity_matrix[i, :])[:k+1]
        adjacency_matrix[i, idx] = similarity_matrix[i, idx]
    # Build edge list
    edge_index = np.array(np.nonzero(adjacency_matrix))
    edge_weight = adjacency_matrix[edge_index[0], edge_index[1]]
    return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_weight, dtype=torch.float)

# Define the ViewEncoder
class ViewEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ViewEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x

# Define the FusionLayer
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

# Define the EVSGL model
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

# Compute target distribution
def target_distribution(q):
    weight = (q ** 2) / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Train the model
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

# Main function
def main():
    # Data paths
    gene_expression_path = 'data/gene/breast_expr_c90.csv'  # Replace with actual gene expression data path
    mri_image_dir = 'data/BRCA_select/'  # Replace with actual MRI image directory

    # Load gene expression data
    gene_expression_df = pd.read_csv(gene_expression_path, index_col=0)

    # Get sample IDs
    sample_ids = gene_expression_df.columns.tolist()

    # Transpose data so that rows are samples and columns are genes
    gene_expression_df = gene_expression_df.transpose()

    # Extract MRI features and get valid sample IDs
    image_features, valid_sample_ids = extract_mri_features(mri_image_dir, sample_ids)

    # Filter gene expression data based on valid_sample_ids
    gene_expression_df = gene_expression_df.loc[valid_sample_ids]
    gene_expression = gene_expression_df.values

    # Ensure data alignment
    assert list(gene_expression_df.index) == valid_sample_ids, "Mismatch in sample IDs after alignment."
    assert gene_expression.shape[0] == image_features.shape[0], "Mismatch in number of samples between gene expression data and MRI features."

    # Build similarity graphs
    edge_index_gene, edge_weight_gene = build_similarity_graph(gene_expression)
    edge_index_image, edge_weight_image = build_similarity_graph(image_features)

    # Convert to tensors
    x_gene = torch.tensor(gene_expression, dtype=torch.float)
    x_image = torch.tensor(image_features, dtype=torch.float)

    # Prepare input lists
    x_list = [x_gene, x_image]
    edge_index_list = [edge_index_gene, edge_index_image]
    edge_weight_list = [edge_weight_gene, edge_weight_image]
    input_dims = [x.shape[1] for x in x_list]

    # Define the model
    hidden_dim = 128
    num_clusters = 5  # Adjust based on your data
    model = EVSGL_Model(input_dims, hidden_dim, num_clusters)

    # Train the model
    fused_rep = train_evsgl(model, x_list, edge_index_list, edge_weight_list, num_clusters)

    # Clustering results
    fused_rep_numpy = fused_rep.detach().numpy()
    kmeans = KMeans(n_clusters=num_clusters, n_init=20)
    y_pred = kmeans.fit_predict(fused_rep_numpy)

    # Output clustering results
    print("Clustering results:", y_pred)

if __name__ == '__main__':
    main()