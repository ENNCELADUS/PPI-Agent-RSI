import math
import torch
import torch.nn.functional as F
from torch.nn import Linear, Embedding, ModuleList, Conv1d, MaxPool1d, ReLU, Sequential
from torch_geometric.nn import global_sort_pool, GCNConv

class DGCNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, max_z, k=0.6, train_dataset=None,
                 dynamic_train=False, GNN=GCNConv, use_feature=False, node_embedding=None):
        super(DGCNN, self).__init__()

        self.use_feature = use_feature
        self.node_embedding = node_embedding

        # ---------- 排序池化 k ----------
        if k <= 1:
            if train_dataset is None:
                k = 30
            else:
                sampled_train = train_dataset[:1000] if dynamic_train else train_dataset
                num_nodes = sorted([g.num_nodes for g in sampled_train])
                k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
                k = max(10, k)
        self.k = 285  # 固定排序池化长度

        # ---------- 嵌入 ----------
        self.max_z = max_z
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        # ---------- 两层 MLP 降维：2560 → 1024 → 512 ----------
        self.input_project = Sequential(
            Linear(2560, 1024),
            ReLU(),
            Linear(1024, 512)
        )

        # ---------- GNN 结构 ----------
        self.convs = ModuleList()
        initial_channels = hidden_channels

        if self.use_feature:
            initial_channels += 512  # 降维后的维度
        if self.node_embedding is not None:
            initial_channels += node_embedding.embedding_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))  # 排序池化用

        # ---------- 1D 卷积 + MLP ----------
        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]

        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, z, edge_index, batch, x=None, edge_weight=None, node_id=None):
        if self.node_embedding is not None and node_id is not None:
            if node_id.max().item() >= self.node_embedding.weight.size(0):
                raise ValueError("Node ID out of bounds")
            if (node_id < 0).any():
                raise ValueError("Negative node_id detected")

        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:
            z_emb = z_emb.sum(dim=1)

        if self.use_feature and x is not None:
            x = self.input_project(x.to(torch.float))  # MLP降维：2560→1024→512
            x = torch.cat([z_emb, x], dim=1)
        else:
            x = z_emb

        if self.node_embedding is not None and node_id is not None:
            n_emb = self.node_embedding(node_id)
            x = torch.cat([x, n_emb], dim=1)

        xs = [x]
        for conv in self.convs:
            xs.append(torch.tanh(conv(xs[-1], edge_index, edge_weight)))
        x = torch.cat(xs[1:], dim=-1)

        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [batch_size, 1, k * feature_dim]

        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
