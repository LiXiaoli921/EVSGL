import torch
import torch.nn.functional as F


def consine_distance(x, y):
    x = x.t()
    y = y.t()
    bs1, bs2 = x.size(0), y.size(0)
    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down
    return 1-cosine


def euclidean_distance(X, Y, square=True):
    x = torch.norm(X, dim=0)
    x = x * x
    x = torch.t(x.repeat(Y.shape[1], 1))

    y = torch.norm(Y, dim=0)
    y = y * y
    y = y.repeat(X.shape[1], 1)

    result = x + y - 2 * (torch.t(X).matmul(Y))
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def probability_graph(X, num_neighbors, dis="euclidean"):

    size = X.shape[1]
    if dis=="cosine":
        distances = consine_distance(X, X)
    else:
        distances = euclidean_distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sort_distances, _ = distances.sort(dim=1)


    topk_distances = sort_distances[:, num_neighbors]
    topk_distances = torch.t(topk_distances.repeat(size, 1)) + 10**-10

    sum_k_distances = torch.sum(sort_distances[:, 0:num_neighbors], dim=1)
    sum_k_distances = torch.t(sum_k_distances.repeat(size, 1))
    sort_distances = None
    torch.cuda.empty_cache()
    T = topk_distances - distances
    distances = None
    torch.cuda.empty_cache()
    probability = torch.div(T, num_neighbors * topk_distances - sum_k_distances)
    T = None
    topk_distances = None
    sum_k_distances = None
    torch.cuda.empty_cache()
    adj = probability.relu().cpu()

    torch.cuda.empty_cache()
    raw_adj = adj
    adj = (adj + adj.t()) / 2
    raw_adj = raw_adj.cuda()
    adj = adj.cuda()
    return adj, raw_adj


def get_Laplacian_from_adj(adj):
    adj_loop = torch.eye(adj.shape[0]).cuda() + adj  #self-loop
    degree = torch.sum(adj_loop, dim=1).pow(-0.5)
    return (adj_loop * degree).t()*degree
