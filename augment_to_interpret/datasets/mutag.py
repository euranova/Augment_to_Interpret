# https://github.com/flyingdoog/PGExplainer/blob/master/MUTAG.ipynb

import yaml
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data

import os


class Mutag(InMemoryDataset):
    def __init__(self, root):
        self.root = str(root)
        super().__init__(root=self.root)
        print(f'Ṕrocessed paths: {self.processed_paths[0]}')
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['Mutagenicity_A.txt', 'Mutagenicity_edge_gt.txt', 'Mutagenicity_edge_labels.txt',
                'Mutagenicity_graph_indicator.txt', 'Mutagenicity_graph_labels.txt', 'Mutagenicity_label_readme.txt',
                'Mutagenicity_node_labels.txt', 'Mutagenicity.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        return 0

    #         msg = f"Dataset should be loaded manually in {self.raw_dir} (see README)."
    #         raise NotImplementedError(msg)

    def process(self):
        with open(Path(self.root, 'Mutagenicity.pkl'), 'rb') as fin:
            _, original_features, original_labels = pkl.load(fin)

        edge_lists, graph_labels, edge_label_lists, node_type_lists = self.get_graph_data()

        data_list = []
        for i in range(original_labels.shape[0]):
            num_nodes = len(node_type_lists[i])
            edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T

            y = torch.tensor(graph_labels[i]).float().reshape(-1, 1)
            x = torch.tensor(original_features[i][:num_nodes]).float()
            assert original_features[i][num_nodes:].sum() == 0
            edge_label = torch.tensor(edge_label_lists[i]).float()
            if y.item() != 0:
                edge_label = torch.zeros_like(edge_label).float()

            node_label = torch.zeros(x.shape[0])
            signal_nodes = list(set(edge_index[:, edge_label.bool()].reshape(-1).tolist()))
            if y.item() == 0:
                node_label[signal_nodes] = 1

            if len(signal_nodes) != 0:
                node_type = torch.tensor(node_type_lists[i])
                node_type = set(node_type[signal_nodes].tolist())
                assert node_type in ({4, 1}, {4, 3}, {4, 1, 3})  # NO or NH

            if y.item() == 0 and len(signal_nodes) == 0:
                continue

            data_list.append(Data(x=x, y=y, edge_index=edge_index, node_label=node_label, edge_label=edge_label,
                                  node_type=torch.tensor(node_type_lists[i])))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_graph_data(self):
        pri = os.path.join(self.root, 'Mutagenicity_')

        file_edges = pri + 'A.txt'
        # file_edge_labels = pri + 'edge_labels.txt'
        file_edge_labels = pri + 'edge_gt.txt'
        file_graph_indicator = pri + 'graph_indicator.txt'
        file_graph_labels = pri + 'graph_labels.txt'
        file_node_labels = pri + 'node_labels.txt'

        edges = np.loadtxt(file_edges, delimiter=',').astype(np.int32)
        edge_labels = np.loadtxt(file_edge_labels, delimiter=',').astype(np.int32)
        graph_indicator = np.loadtxt(file_graph_indicator, delimiter=',').astype(np.int32)
        graph_labels = np.loadtxt(file_graph_labels, delimiter=',').astype(np.int32)
        node_labels = np.loadtxt(file_node_labels, delimiter=',').astype(np.int32)

        graph_id = 1
        starts = [1]
        node2graph = {}
        for i in range(len(graph_indicator)):
            if graph_indicator[i] != graph_id:
                graph_id = graph_indicator[i]
                starts.append(i + 1)
            node2graph[i + 1] = len(starts) - 1
        # print(starts)
        # print(node2graph)
        graphid = 0
        edge_lists = []
        edge_label_lists = []
        edge_list = []
        edge_label_list = []
        for (s, t), l in list(zip(edges, edge_labels)):
            sgid = node2graph[s]
            tgid = node2graph[t]
            if sgid != tgid:
                print('edges connecting different graphs, error here, please check.')
                print(s, t, 'graph id', sgid, tgid)
                exit(1)
            gid = sgid
            if gid != graphid:
                edge_lists.append(edge_list)
                edge_label_lists.append(edge_label_list)
                edge_list = []
                edge_label_list = []
                graphid = gid
            start = starts[gid]
            edge_list.append((s - start, t - start))
            edge_label_list.append(l)

        edge_lists.append(edge_list)
        edge_label_lists.append(edge_label_list)

        # node labels
        node_label_lists = []
        graphid = 0
        node_label_list = []
        for i in range(len(node_labels)):
            nid = i + 1
            gid = node2graph[nid]
            # start = starts[gid]
            if gid != graphid:
                node_label_lists.append(node_label_list)
                graphid = gid
                node_label_list = []
            node_label_list.append(node_labels[i])
        node_label_lists.append(node_label_list)

        return edge_lists, graph_labels, edge_label_lists, node_label_lists


class MutagNoise(Mutag):
    def __init__(self, root, node_features):
        super().__init__(root=root)

        self.data, self.slices = torch.load(self.processed_paths[0])
        if 'n_f_constant' not in node_features:
            node_features['n_f_constant'] = 0
        if 'n_f_binary' not in node_features:
            node_features['n_f_binary'] = 0
        if 'n_f_gaussian' not in node_features:
            node_features['n_f_gaussian'] = 0
        self.node_features = node_features
        N = self.data['x'].shape[0]
        gausian_noise = torch.randn(N, node_features['n_f_gaussian']).float()
        constant_noise = torch.cat([
            torch.tensor(np.random.uniform(low=0, high=1, size=N)).view(-1, 1).float()
            for _ in range(node_features['n_f_constant'])
        ], dim=1) if node_features['n_f_constant'] > 0 else torch.tensor([])
        binary_noise = torch.cat([
            torch.tensor((np.random.uniform(low=0, high=1, size=N) >= 0.5).astype(int)).view(-1, 1).float()
            for _ in range(node_features['n_f_constant'])
        ], dim=1) if node_features['n_f_constant'] > 0 else torch.tensor([])
        self.data['x'] = torch.cat([self.data['x'][:, node_features['original']],
                                    binary_noise, constant_noise, gausian_noise,
                                    ], dim=1)
