#!/usr/bin/python
# coding=utf-8
import xlrd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

sex = 0
height = 1
weight = 2
is_like_math = 3
is_like_liter = 4
is_like_sport = 5
layer = [5, 5, 1]
layer_len = len(layer)
err = 0.1
step = 1
weight_map = {}
out = {}


def get_row(s, ii):
    # 获取表格中的数据,并将填充其中的空数据.
    d = list(s.row_values(ii, 1, 9))
    d.pop(1)
    d.pop(3)
    for jj in range(0, d.count('')):
        index = d.index('')
        d.pop(index)
        d.insert(index, 0)
    return d


def fp(dd, ii):
    global out
    if ii == layer_len:
        out[ii - 1] = np.array(dd)
        return dd
    mm = weight_map[ii]
    out[ii - 1] = np.array(dd)
    # next_dd = np.matrix([(np.array(m[i, :]) * np.array(dd)).tolist()[0] for i in range(0, n)])
    next_dd = np.array([np.sum(np.array(mm[jj, :]) * np.array(dd)) for jj in range(0, mm.shape[0])])
    # next_dd = np.reshape(next_dd, (layer[ii], layer[ii - 1]))
    # next_dd = next_dd.sum(axis=1).T
    # next_dd=np.array(next_dd)
    next_dd = 1 / (1 + np.exp(-next_dd))
    return fp(next_dd, ii + 1)


def bp(weight_mat, mid_output, delta, ii, final_out=None):
    global weight_map
    if ii == 0:
        return
    elif ii == layer_len - 1:
        delta = mid_output[ii] * (1 - mid_output[ii]) * (final_out - mid_output[ii])
        dw = step * np.matrix(delta).T * np.matrix(mid_output[ii - 1])
        weight_mat = weight_map[ii]
        weight_map[ii] = dw + weight_map[ii]
        bp(weight_mat, mid_output, delta, ii - 1)
    else:
        delta = mid_output[ii] * (1 - mid_output[ii]) * np.array(np.matrix(delta) * np.matrix(weight_mat))
        dw = step * np.matrix(delta).T * np.matrix(mid_output[ii - 1])
        weight_mat = weight_map[ii]
        weight_map[ii] = dw + weight_map[ii]
        bp(weight_mat, mid_output, delta, ii - 1)


workbook = xlrd.open_workbook('../homeworkData_2018.xls')
sheet = workbook.sheet_by_index(0)
N = sheet.nrows - 1

for i in range(1, len(layer)):
    weight_map[i] = np.matrix(np.random.random((layer[i], layer[i - 1])))
data = [get_row(sheet, i) for i in range(1, N + 1)]
data_len = len(data)
i = 0
# 去重
while i < data_len:
    if data.count(data[i]) > 1:
        data.pop(i)
        data_len = len(data)
    i = i + 1
# 归一化
data = np.matrix(data)
data[:, 1:] = scale(data[:, 1:])

i = 0
data_len = len(data)
err_sum = err * data_len + 1
err_list = []
# 开始迭代
while err_sum > err * data_len:
    i += 1
    r = fp(data[i % data_len, 1:], 1)
    bp(weight_map[layer_len - 1], out, 0, layer_len - 1, final_out=data[i % data_len, 0])
    err_sum = [np.abs(fp(data[j, 1:], 1) - data[j, 0]) for j in range(data_len)]
    if i % 100 == 0:
        print(i, list(np.array(err_sum) > 0.5).count(True), np.sum(err_sum))
    err_sum = sum(err_sum)
    err_list.append(err_sum)

m = [np.abs(fp(data[j, 1:], 1) - data[j, 0]) > 0.5 for j in range(data_len)]
print(m.count(True))
plt.plot(err_list)
plt.show()
# print([np.abs(fp(data[j, 1:], 1) - data[j, 0]) for j in range(len(data))])
