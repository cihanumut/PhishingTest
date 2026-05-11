import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np

class GNNModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=4)
        self.conv2 = GATConv(hidden_channels * 4, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphModels:
    def __init__(self, num_features):
        self.num_features = num_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GNNModel(num_features, 32, 2).to(self.device)

    def build_graph(self, X, y, sample_size=10000):
        """
        Builds a bipartite graph where nodes are samples and feature-values.
        """
        print(f"Building bipartite graph for {sample_size} samples...")
        X_sub = X[:sample_size]
        y_sub = y[:sample_size]
        
        # We'll use a Sample-Sample graph based on a more robust similarity for now
        # because a full bipartite graph with 111 features would be huge.
        # But we'll improve the similarity metric and use GAT.
        from sklearn.neighbors import kneighbors_graph
        # Link each node to its 5 nearest neighbors
        adj = kneighbors_graph(X_sub, n_neighbors=5, mode='connectivity', include_self=False)
        
        edges = adj.tocoo()
        edge_index = torch.tensor(np.array([edges.row, edges.col]), dtype=torch.long)
        
        x = torch.tensor(X_sub, dtype=torch.float)
        y_tensor = torch.tensor(y_sub, dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y_tensor)

    def train(self, data, epochs=100):
        print(f"Training GAT model on {self.device}...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-4)
        self.model.train()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
            loss = F.nll_loss(out, data.y.to(self.device))
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f'Epoch {epoch}: Loss {loss.item():.4f}')

    def evaluate(self, data):
        self.model.eval()
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        pred = out.argmax(dim=1)
        correct = (pred == data.y.to(self.device)).sum()
        acc = int(correct) / int(data.y.size(0))
        print(f'GAT Accuracy: {acc:.4f}')
        return acc
