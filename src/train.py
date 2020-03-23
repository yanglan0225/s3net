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
num_epoches = 1
learning_rate = 0.001
data_dir = '/home/yl/sketchrnn.txt'
class_list = '/home/yl/sketchrnn.txt'
theta_list = np.load('/home/yl/theta.npy')
train_model_save_dir = '/home/yl/data/train_model/s3net'
save_dir = '/home/yl/data/model/s3net.pkl'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
print('Train model save dir:  ', train_model_save_dir)
print('Final model save path:  ', save_dir)


"""
    dataset and data loader
"""
print('='*10, 'Start Data Loading', '='*10)
train_dataset = data_loader.QuickDraw(data_dir, class_list, theta_list, type='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = data_loader.QuickDraw(data_dir, class_list, theta_list, type='valid')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print('='*10, 'Data Loaded', '='*10)


"""
    model and optimizer
"""

model = model.Net(input_chanel, hidden_chanel, fea_dim, hidden_chanel2, hidden_chanel3, out_chanel, num_class, n_rnn_layer).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.load_state_dict(torch.load('/home/yl/data/train_model/s3net/4.pkl'))


train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
best_acc = 0.0

print('='*10, 'Start training', '='*10)

for epoch in range(num_epoches):
    # print('=' * 10, 'Epoch ', epoch,  '=' * 10)
    # if epoch == 5:
    #     optimizer.param_groups[0]['lr'] = 1e-4
    # if epoch == 10:
    #     optimizer.param_groups[0]['lr'] = 1e-5
    # if epoch == 15:
    #     optimizer.param_groups[0]['lr'] = 1e-6
    print('learning rate: ', optimizer.param_groups[0]['lr'])

    since = time.time()
    running_acc = 0.0
    running_loss = 0.0
    val_loss = 0.0
    val_acc = 0.0
    model.train()
    for i, data in enumerate(train_loader):
        inputs = data
        label = data['y'].to(device).long()
        inputs = inputs.to(device)
        optimizer.zero_grad()
        output, prediction, link_loss, ent_loss = model(inputs)
        loss = F.nll_loss(output, label.view(-1)) + link_loss + ent_loss
        loss.backward()
        running_loss += data.y.size(0) * loss.item()
        optimizer.step()
        _, preds = torch.max(output, 1)
        running_acc += torch.sum(preds == label.data)
        if i % 10 == 0:
            print('the {}-th batch, loss: {:.6f}, acc: {:.6f}'.format(i, running_loss / (i*inputs.num_graphs + 1),
                                                                      running_acc.double().cpu() / (i*inputs.num_graphs + 1)))
    #return loss_all / len(train_dataset)
    j = running_loss / (len(train_dataset))
    e = running_acc.double().cpu() / (len(train_dataset))
    print('Finish {} epoch, Loss:{:.6f}, Acc:{:.6f}'.format(epoch + 1, j, e))
    train_loss.append(j)
    train_acc.append(e)
    time_epoch = time.time() - since
    print("This epoch train costs time:{:.0f}m {:.0f}s".format(time_epoch // 60, time_epoch % 60))

    model.eval()
    loss = 0.0
    for i, data in enumerate(val_loader):
        inputs = data
        label = data['y'].to(device).long()
        inputs = inputs.to(device)
        output, prediction, link_loss, ent_loss = model(inputs)
        loss = F.nll_loss(output, label.view(-1)) + link_loss + ent_loss
        val_loss = val_loss + data.y.size(0) * loss.item()
        _, preds = torch.max(output, 1)
        val_acc += torch.sum(preds == label.data)
        d = val_acc.double().cpu()
    save_path = os.path.join(train_model_save_dir, str(epoch) + '.pkl')
    torch.save(model.state_dict(), save_path)
    c = val_loss / (len(val_dataset))
    f = d / (len(val_dataset))
    if f > best_acc:
        best_acc = f
        best_model_wts = copy.deepcopy(model.state_dict())
    print('val: Loss:{:.6f}, Acc:{:.6f}'.format(c, f))
    valid_loss.append(c)
    valid_acc.append(f)
    time_epoch_val = time.time() - since
    del c, d, f
    print("This epoch val costs time:{:.0f}m {:.0f}s".format(time_epoch_val // 60, time_epoch_val % 60))


model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), save_dir)
print('train_loss:{} train_acc:{} val_loss{} val_acc{}'.format(train_loss, train_acc, valid_loss, valid_acc))
