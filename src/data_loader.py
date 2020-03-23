import os
import len_group
import numpy as np
import torch
import torch.utils.data as data
from torch_geometric.data import Data


"""
    Define the device
"""
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

class QuickDraw(data.Dataset):
    def __init__(self, data_dir, class_list, theta_list, type):
        """

        :param data_dir: txt file, the path of sketches
        :param class_list: txt file, the path of category info
        :param theta_list: numpy file, save theta for each category
        :param type: 'train', 'vaild', 'test'
        """
        self.class_list = class_list
        self.type = type
        self.classes, self.class_to_idx = self.find_class(class_list)
        self.label_data_npy = np.zeros((1, 2)) # initial the array to save the label and sketches, dim0 is label, dim1 is storke-3 sketch
        self.theta_list = theta_list
        with open(data_dir) as class_url_list:
            for classes_list in class_url_list:

                self.classnpy = np.load(classes_list.replace('yanglan', 'yl/data').strip(), encoding='latin1', allow_pickle=True)
                classpath1, tempclass = os.path.split(classes_list)
                classname, exten = os.path.splitext(tempclass)
                self.label = self.class_to_idx[classname]

                if self.type == 'train':
                     np.random.shuffle(self.classnpy['train'])
                     self.coordinate = self.classnpy['train'][:9000]
                     self.label_np = self.label * np.ones((9000, 1))
                     label_data_npy = np.c_[self.label_np, self.coordinate.reshape(9000, -1)]
                     self.label_data_npy = np.r_[self.label_data_npy, label_data_npy]

                if self.type == 'valid':
                     self.coordinate = self.classnpy['valid']
                     self.label_np = self.label * np.ones((2500, 1))
                     label_data_npy = np.c_[self.label_np, self.coordinate.reshape(2500, -1)]
                     self.label_data_npy = np.r_[self.label_data_npy, label_data_npy]

                if self.type == 'test':
                     self.coordinate = self.classnpy['test']
                     self.label_np = self.label * np.ones((2500, 1))
                     label_data_npy = np.c_[self.label_np, self.coordinate.reshape(2500, -1)]
                     self.label_data_npy = np.r_[self.label_data_npy, label_data_npy]

        self.label_data_npy1 = self.label_data_npy[1:, :] # remove the first useless element
        self.max_length, self.max_groupnum = self.max_len()




    def __len__(self):
        return len(self.label_data_npy1)


    def __getitem__(self, item):
        tempcoordinate = self.label_data_npy1[item]
        label = tempcoordinate[0]
        coordinate = tempcoordinate[1] # original coordinate
        coordinate2 = np.zeros((self.max_length, 3))
        coordinate2[:len(coordinate)] = coordinate
        c = torch.from_numpy(coordinate2).to(device)
        groupid = len_group.get_group(coordinate, self.theta_list[int(label)])
        src, dst, groupNum = self.get_affinity_matrix(torch.squeeze(torch.from_numpy(groupid)))
        edge_idx = torch.tensor([np.concatenate((src,dst)), np.concatenate((dst,src))],dtype=torch.long)
        feature = torch.zeros((len(coordinate)+1, 128))
        data = Data(x=feature, edge_index=edge_idx, y=label, s=len(coordinate), g=int(groupNum.item()), c=c)
        del feature, edge_idx


        return data


    def find_class(self, dir):
        with open(dir) as class_url_list:
            classlist = []
            for classpath in class_url_list:
                classpath1, tempclass = os.path.split(classpath)
                classname, exten = os.path.splitext(tempclass)
                classlist.append(classname)
        classlist.sort()
        class_to_idx = {classlist[i]: i for i in range(len(classlist))}
        return classlist, class_to_idx

    def max_len(self):
        max_len = 0
        pos = 0
        for i in range(len(self.label_data_npy1)):

            if len(self.label_data_npy1[i][1]) >= max_len:
                max_len = len(self.label_data_npy1[i][1])
                pos = i
        groupid = len_group.get_group(self.label_data_npy1[pos][1], self.theta_list[int(self.label_data_npy1[pos][0])])
        src, dst, groupNum = self.get_affinity_matrix(torch.squeeze(torch.from_numpy(groupid)))
        return max_len, groupNum

    def get_affinity_matrix(self, groupId):
        groupnum = torch.max(groupId)
        src = []
        dst = []
        repre_point = []
        id = 0

        # select the first point of each stroke as the representative point
        for i in range(len(groupId)):
            if i == 0:
                repre_point.append(i)
                id = groupId[i][0]
            else:
                if groupId[i][0] != id: # next group
                    repre_point.append(i)
                    id = groupId[i][0]
        repre_point.append(len(groupId)-1)


        # build edges of rule 1
        for i in range(len(repre_point)-1):
            for j in range(repre_point[i]+1, repre_point[i+1]+1):
                src.append(i)
                dst.append(j)


        # build edges of rule 2
        for i in range(len(repre_point)-1):
            if groupId[repre_point[i]][1] == groupId[repre_point[i+1]][1]:
                src.append(repre_point[i])
                dst.append(repre_point[i+1])


        # build edges of rule 3
        for i in range(len(repre_point)-1):
            src.append(len(groupId))
            dst.append(repre_point[i])


        return np.array(src), np.array(dst), groupnum + 1



    def abs_data(self, data):
        abs_x = np.zeros(len(data))
        abs_y = np.zeros(len(data))
        abs_x[0] = data[0][0]
        abs_y[0] = data[0][1]
        ## convert the relative corrinates to the absolute corrdinates
        result = np.zeros((len(data), 3))
        for i in range(len(data)):
            if i != 0:
                abs_x[i] = abs_x[i - 1] + data[i][0]
                abs_y[i] = abs_y[i - 1] + data[i][1]

        min_x = np.min(abs_x)
        min_y = np.min(abs_y)
        max_x = np.max(abs_x)
        max_y = np.max(abs_y)
        normalize_factor = np.max((max_x - min_x, max_y - min_y))
        result[:, 0] = abs_x / normalize_factor
        result[:, 1] = abs_y / normalize_factor
        # np.divide(data[:,0], normalize_factor)
        result[:, 2] = data[:, 2]
        # data = ori_data
        return result
