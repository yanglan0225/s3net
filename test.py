import torch
import torch.nn.functional as F
import data_loader
from torch_geometric.data import DataLoader
import numpy as np
import time
import copy
import os
import model


max_nodes = 400
batch_size = 250
num_class  = 345
input_chanel = 3
hidden_chanel = 512
fea_dim = 128
hidden_chanel2 = 256
hidden_chanel3 = 512
out_chanel = 1024
n_rnn_layer = 2
num_epoches = 20
learning_rate = 0.001
data_dir = '/home/yl/sketchrnn.txt'
class_list = '/home/yl/sketchrnn.txt'
theta_list = np.load('/home/yl/theta.npy')
model_path = '/home/yl/data/train_model/s3net/5.pkl'
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

print('='*10, 'Initial Setting', '='*10)
print('Batch Size:  ', batch_size)
print('Data_dir:  ', data_dir)
print('Input dim:  ', input_chanel)
print('hidden dim:  ', hidden_chanel, ' ', hidden_chanel2, ' ', hidden_chanel3)
print('Output dim:  ', out_chanel)
print('RNN Layers:  ', n_rnn_layer)
print('Num epochs:', num_epoches)
print('Learning rate: ', learning_rate)
print('Data_dir :', data_dir)
print('Class info:  ', class_list)
print('Device: ', device)
print('Train model save dir:  ', model_path)



"""
    dataset and data loader
"""
print('='*10, 'Start Data Loading', '='*10)
test_dataset = data_loader.QuickDraw(data_dir, class_list, theta_list, type='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print('='*10, 'Data Loaded', '='*10)


"""
    model 
"""

model = model.Net(input_chanel, hidden_chanel, fea_dim, hidden_chanel2, hidden_chanel3, out_chanel, num_class, n_rnn_layer).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

"""
    test procedure
"""

model.eval()
test_acc = 0.0
test_loss = 0.0
loss = 0.0

for i, data in enumerate(test_loader):
    inputs = data
    label = data['y'].to(device).long()
    inputs = inputs.to(device)
    with torch.no_grad():
        output, prediction, link_loss, ent_loss = model(inputs)
        loss = F.nll_loss(output, label.view(-1)) + link_loss + ent_loss
        test_loss = test_loss + data.y.size(0) * loss.item()
        _, preds = torch.max(output, 1)
        test_acc += torch.sum(preds == label.data)
        e = test_acc.double().cpu()


g = test_loss / (len(test_dataset))
h = e / (len(test_dataset))
print('test: Loss:{:.6f}, Acc:{:.6f}'.format(g, h))
