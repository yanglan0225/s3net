import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch.autograd import Variable
import torch_geometric.utils as utils



device = torch.device('cuda:2')

class GNN(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 normalize=False,
                 add_loop=False,
                 lin=True):
        super(GNN, self).__init__()

        self.add_loop = add_loop

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask, self.add_loop)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask, self.add_loop)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask, self.add_loop)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self, input_chanel, hidden_chanel, fea_dim, hidden_chanel2,hidden_chanel3, out_chanel, num_class, n_rnn_layer):
        super(Net, self).__init__()

        num_nodes = 5

        self.gnn1_pool = GNN(fea_dim, hidden_chanel2, num_nodes)
        self.gnn1_embed = GNN(fea_dim, hidden_chanel2, hidden_chanel2, lin=False)

        self.gnn3_embed = GNN(3 * hidden_chanel2, hidden_chanel3, out_chanel, lin=False)

        self.lin1 = torch.nn.Linear(2 * hidden_chanel3 + out_chanel, out_chanel)
        self.lin2 = torch.nn.Linear(out_chanel, num_class)
        self.n_layer = n_rnn_layer
        self.n_classes = num_class
        self.lstm = nn.LSTM(input_chanel, hidden_chanel, n_rnn_layer, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_chanel * 2, fea_dim)
        self.classify = nn.Linear(out_chanel, num_class)
        self.fea_dim = fea_dim


    def forward(self, data):
        seq_len = data['s']

        inputs = data['c'].reshape((len(seq_len), -1, 3))

        inputs = inputs.reshape((len(seq_len), -1, 3))
        _, idx_sort = torch.sort(seq_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        input_x = inputs.index_select(0, Variable(idx_sort))
        length_list = list(seq_len[idx_sort])
        input_x = input_x.float()
        pack = nn.utils.rnn.pack_padded_sequence(input_x, length_list, batch_first=True)
        out, state = self.lstm(pack)
        del state
        un_padded = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        un_padded = un_padded[0].index_select(0, Variable(idx_unsort))
        out = self.dropout(un_padded)
        feature = self.fc(out)
        batch_feature = None
        del out, pack, un_padded
        for i in range(data.num_graphs):
            emptyfeature = torch.zeros((1, self.fea_dim)).to(device)
            fea = torch.cat((feature[i][:(seq_len[i])], emptyfeature))
            if batch_feature is None:
                batch_feature = fea
            else:
                batch_feature = torch.cat((batch_feature, fea))

        data['x'] = batch_feature
        x, edge_index = data.x, data.edge_index
        dense_x = utils.to_dense_batch(x, batch=data.batch)
        x = dense_x[0]
        adj = utils.to_dense_adj(data.edge_index, batch=data.batch)
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)
        x, adj, l1, e1 = dense_diff_pool(x, adj, s)


        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x1 = self.lin1(x)
        x = F.relu(x1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), x1, l1, e1

