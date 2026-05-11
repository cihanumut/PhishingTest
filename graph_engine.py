import networkx as nx
import pandas as pd
import numpy as np

class GraphEngine:
    def __init__(self):
        self.G = nx.Graph()

    def build_feature_graph(self, df, important_cols=None):
        """
        Creates a graph based on relations between high-impact features.
        """
        print("Building feature-relation graph...")
        if important_cols is None:
            important_cols = ['qty_dot_url', 'length_url', 'domain_length', 'qty_params', 'tls_ssl_certificate']
        
        # We'll use a subset of samples to build the relation graph to avoid memory blowup,
        # then we map these relations back to all samples.
        subset = df.sample(min(10000, len(df)))
        
        for _, row in subset.iterrows():
            nodes = []
            for col in important_cols:
                node_id = f"{col}_{row[col]}"
                nodes.append(node_id)
            
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if self.G.has_edge(nodes[i], nodes[j]):
                        self.G[nodes[i]][nodes[j]]['weight'] += 1
                    else:
                        self.G.add_edge(nodes[i], nodes[j], weight=1)

    def calculate_metrics(self):
        print("Calculating Degree Centrality...")
        centrality = nx.degree_centrality(self.G)
        return centrality

    def augment_features(self, df, important_cols=None):
        """
        Adds individual Centrality metrics for each important feature to the dataframe.
        """
        if important_cols is None:
            important_cols = ['qty_dot_url', 'length_url', 'domain_length', 'qty_params', 'tls_ssl_certificate']
            
        cent = self.calculate_metrics()
        
        print("Augmenting dataset with granular graph metrics...")
        
        for col in important_cols:
            col_name = f'cent_{col}'
            df[col_name] = df[col].apply(lambda x: cent.get(f"{col}_{x}", 0))
        
        return df
