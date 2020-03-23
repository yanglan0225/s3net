import numpy as np


def abs_data(data):
    abs_x = np.zeros(len(data))
    abs_y = np.zeros(len(data))
    abs_x[0] = data[0][0]
    abs_y[0] = data[0][1]
    ## convert the relative corrinates to the absolute corrdinates
    result = np.zeros((len(data),3))
    for i in range(len(data)):
        if i != 0 :

            abs_x[i] = abs_x[i-1] + data[i][0]
            abs_y[i] = abs_y[i-1] + data[i][1]

    min_x = np.min(abs_x)
    min_y = np.min(abs_y)
    max_x = np.max(abs_x)
    max_y = np.max(abs_y)
    normalize_factor = np.max((max_x-min_x, max_y-min_y))
    result[:,0] = abs_x/normalize_factor
    result[:,1] = abs_y/normalize_factor
    #np.divide(data[:,0], normalize_factor)
    result[:,2] = data[:,2]
    #data = ori_data
    return result

def get_group(data, theta):
    absdata = abs_data(data)
    group_idx = 0
    length = 0
    label = 0
    stroke_id = 0
    group_result = np.zeros((len(data), 2), dtype=np.int)
    for i in range((len(data)-1)):
        if data[i][2] == 1:
            group_result[i, 1] = stroke_id
            stroke_id += 1
            dis = np.sqrt(np.sum(np.power(absdata[i + 1] - absdata[i], 2)[:2]))
            if dis >= 0.3 * theta:
                group_result[group_idx:(i + 1), 0] = label
                group_idx = i + 1
                label += 1
                length = 0
        else:
            group_result[i, 1] = stroke_id
            length += np.sqrt(np.sum(np.power(absdata[i+1] - absdata[i], 2)[:2]))
            if length >= theta:
                group_result[group_idx:(i+1), 0] = label
                group_idx = i+1
                label += 1
                length = 0

    group_result[group_idx:, 0] = label
    group_result[-1, 1] = stroke_id
    a = group_result.astype(int)
    return a








