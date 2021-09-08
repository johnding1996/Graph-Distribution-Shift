"""
    File to load dataset based on user control from main file
"""
import torch
from data.superpixels import SuperPixDGL

# DATASET_NAME = 'MNIST'
# DATASET_NAME = 'CIFAR10'
DATASET_NAME = 'RotatedMNIST'


def data_list(name):
    data_dir = '/cmlscratch/kong/projects/Domain-Transfer-Graph/preprocessing/superpixel/data'
    dataset = SuperPixDGL(data_dir=data_dir, dataset=name, split='train')

    graph_lists = dataset.graph_lists
    graph_labels = dataset.graph_labels

    data_list = []
    for G, y in zip(graph_lists, graph_labels):
        y = y.to(torch.long)
        x = G.ndata['feat'].to(torch.float)
        row, col = G.edges()
        edge_index = torch.stack([row, col], dim=0).to(torch.long)
        data = {'x': x, 'y': y, 'edge_index': edge_index}
        data_list.append(data)

    return data_list


if __name__ == "__main__":
    data_list = data_list(DATASET_NAME)
    torch.save(data_list,
               f'/cmlscratch/kong/projects/Domain-Transfer-Graph/preprocessing/superpixel/data/{DATASET_NAME}.pt')

    # dataset = SuperPixDataset('MNIST')
    # data_list = torch.load(f'/cmlscratch/kong/projects/Domain-Transfer-Graph/preprocessing/superpixel/data/{DATASET_NAME}.pt')

    pdb.set_trace()
