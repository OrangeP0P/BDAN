import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def cal_tradeoff():
    channel_location_L = pd.read_excel('Datasets_Transfer_Task/OpenBMI/PermutatedData/ch_loc_L.xls')
    Xe = channel_location_L.X
    Ye = channel_location_L.Y
    ET_L = np.ones([len(Xe), len(Ye)])
    for i in range(0, len(Xe)):
        for j in range(0, len(Ye)):
            ET_L[i, j] = math.sqrt((Xe[i] - Xe[j]) * (Xe[i] - Xe[j]) + (Ye[i] - Ye[j]) * (Ye[i] - Ye[j]))
    np.save('Datasets_Transfer_Task/OpenBMI/PermutatedData/ET_L.npy', ET_L)

    channel_location_R = pd.read_excel('Datasets_Transfer_Task/OpenBMI/PermutatedData/ch_loc_R.xls')
    Xe = channel_location_R.X
    Ye = channel_location_R.Y
    ET_R = np.ones([len(Xe), len(Ye)])
    for i in range(0, len(Xe)):
        for j in range(0, len(Ye)):
            ET_R[i, j] = math.sqrt((Xe[i] - Xe[j]) * (Xe[i] - Xe[j]) + (Ye[i] - Ye[j]) * (Ye[i] - Ye[j]))
    np.save('Datasets_Transfer_Task/OpenBMI/PermutatedData/ET_R.npy', ET_R)

    region_location = pd.read_excel('Datasets_Transfer_Task/OpenBMI/PermutatedData/reg_loc.xls')
    Xe = region_location.X
    Ye = region_location.Y
    RT = np.ones([len(Xe), len(Ye)])
    for i in range(0, len(Xe)):
        for j in range(0, len(Ye)):
            RT[i, j] = math.sqrt((Xe[i] - Xe[j]) * (Xe[i] - Xe[j]) + (Ye[i] - Ye[j]) * (Ye[i] - Ye[j]))
    np.save('Datasets_Transfer_Task/OpenBMI/PermutatedData/RT.npy', RT)


def get_centerpoint(lis):
    area = 0.0
    x, y = 0.0, 0.0

    a = len(lis)
    for i in range(a):
        lat = lis[i][0]  # weidu
        lng = lis[i][1]  # jingdu

        if i == 0:
            lat1 = lis[-1][0]
            lng1 = lis[-1][1]

        else:
            lat1 = lis[i - 1][0]
            lng1 = lis[i - 1][1]

        fg = (lat * lng1 - lng * lat1) / 2.0

        area += fg
        x += fg * (lat + lat1) / 3.0
        y += fg * (lng + lng1) / 3.0

    x = x / area
    y = y / area

    return x, y

# channel_location = pd.read_excel('Datasets_Transfer_Task/OpenBMI/PermutatedData/ch_loc.xls')
# np.save('channel_location.npy', channel_location)
# Xe = channel_location.X
# Ye = channel_location.Y
# ET = np.ones([len(Xe), len(Ye)])
# for i in range(0, len(Xe)):
#     for j in range(0, len(Ye)):
#         ET[i, j] = math.sqrt((Xe[i] - Xe[j]) * (Xe[i] - Xe[j]) + (Ye[i] - Ye[j]) * (Ye[i] - Ye[j]))
# np.save('ET.npy', ET)

# channel = 56
# E1 = 8
# S1 = 5
# E2 = 5
# S2 = 5
#
#
# channel_location = np.load('channel_location.npy', allow_pickle=True)
# START_ELE, END_ELE = [], []
# X_loc, Y_loc = [], []
# loc_num = 0
# for ele in range(0, channel, S1):  # 依次读取区域
#     if ele < channel - E1:
#         loc_num = loc_num + 1
#         location_size = E1  # 电极区域大小
#         start_ele = ele
#         end_ele = start_ele + location_size  # 区域终止电极编号
#         list.append(START_ELE, start_ele)
#         list.append(END_ELE, end_ele)
#     elif ele < channel - S1:  # 最后一个不完全区域
#         loc_num = loc_num + 1
#         location_size = channel - ele  # 电极区域大小
#         start_ele = ele
#         end_ele = channel  # 区域终止电极编号
#         list.append(START_ELE, start_ele)
#         list.append(END_ELE, end_ele)
#
# for loc in range(0, loc_num):
#     loc_list = channel_location[START_ELE[loc]:END_ELE[loc],:]
#     inside_loc_list = loc_list[0:S1]
#     conter_loc_list = loc_list[S1:E1]
#     x_loc, y_loc = np.mean(loc_list[:, 0]), np.mean(loc_list[:, 1])  # 求解区域的质心
#     list.append(X_loc, x_loc)
#     list.append(Y_loc, y_loc)
#
# RT = np.ones([loc_num, loc_num])
# for i in range(0, E2):
#     for j in range(0, E2):
#         RT[i, j] = math.sqrt((X_loc[i] - X_loc[j]) * (X_loc[i] - X_loc[j]) + (Y_loc[i] - Y_loc[j]) * (Y_loc[i] - Y_loc[j]))
#
# plt.figure()
# color_list = ['blue', 'darkorange', 'steelblue', 'green', 'purple',
#               'coral', 'cyan', 'steelblue', 'plum', 'tan', 'slateblue', 'blue']  # 颜色列表
# for loc in range(0, loc_num):
#     loc_list = channel_location[START_ELE[loc]:END_ELE[loc],:]
#     inside_loc_list = loc_list[0:S1]
#     conter_loc_list = loc_list[S1:E1]
#     plt.scatter(inside_loc_list[:,0], inside_loc_list[:,1], s = 10, c=color_list[loc])
#     plt.scatter(conter_loc_list[:,0], conter_loc_list[:,1], s = 10, alpha=0.8, linewidths=5,
#                 edgecolors=color_list[loc], c=color_list[loc+1])
#     x_loc, y_loc = np.mean(loc_list[:,0]), np.mean(loc_list[:,1])  # 求解区域的质心
#     plt.scatter(x_loc,y_loc, s = 150, linewidths=2, edgecolors='black', c=color_list[loc])
# plt.show()

# channel_location = np.load('channel_location.npy', allow_pickle=True)
# START_ELE, END_ELE = [], []
# X_loc, Y_loc = [], []
# for ele in range(0, channel, S1):  # 依次读取区域
#     if ele < channel - E1:
#         location_size = E1  # 电极区域大小
#         start_ele = ele
#         end_ele = start_ele + location_size  # 区域终止电极编号
#         list.append(START_ELE, start_ele)
#         list.append(END_ELE, end_ele)
#
# for loc in range(0, E2):
#     loc_list = channel_location[START_ELE[loc]:END_ELE[loc],:]
#     inside_loc_list = loc_list[0:S1]
#     conter_loc_list = loc_list[S1:E1]
#     x_loc, y_loc = np.mean(loc_list[:, 0]), np.mean(loc_list[:, 1])  # 求解区域的质心
#     list.append(X_loc, x_loc)
#     list.append(Y_loc, y_loc)
#
#
# RT = np.ones([E2, E2])
# for i in range(0, E2):
#     for j in range(0, E2):
#         RT[i, j] = math.sqrt((X_loc[i] - X_loc[j]) * (X_loc[i] - X_loc[j]) + (Y_loc[i] - Y_loc[j]) * (Y_loc[i] - Y_loc[j]))
#
# # color_list = ['blue', 'yellow', 'green', 'black', 'pink', 'blue', 'yellow', 'green', 'black', 'pink']
# # for loc in range(0, np.size(START_ELE)):
# #     loc_list = channel_location[START_ELE[loc]:END_ELE[loc],:]
# #     inside_loc_list = loc_list[0:S1]
# #     conter_loc_list = loc_list[S1:E1]
# #     plt.scatter(inside_loc_list[:,0], inside_loc_list[:,1], s = 10, c=color_list[loc])
# #     plt.scatter(conter_loc_list[:,0], conter_loc_list[:,1], s = 30, c=color_list[1])
# #     x_loc, y_loc = get_centerpoint(loc_list)
# #     plt.scatter(x_loc,y_loc,c='red')
