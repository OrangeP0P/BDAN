import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
from TSNEDataloader import read_test_data
from torch.autograd import Variable
import scipy.io as io
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.decomposition import PCA

'''提取区域1的电极特征'''
def feature_ele_r1(subject, ELE_Net, channel_region):
    ELE_Net_Electrode = nn.Sequential(*list(ELE_Net.children()))[:1]
    test_data_L, test_data_R, test_label, te_num = read_test_data(Basic_folder, Current_datasets, subject)
    test_data = Data.TensorDataset(torch.from_numpy(test_data_L.astype(np.float32)),
                                   torch.from_numpy(test_data_R.astype(np.float32)),
                                   torch.from_numpy(test_label.astype(np.float32)))

    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    '''电极特征-aa'''

    for data_L, data_R, label in test_loader:
        data_L, data_R, label = data_L.cuda(), data_R.cuda(), label.cuda()
        data_L, data_R, label = Variable(data_L), Variable(data_R), Variable(label)
        feature_L = ELE_Net_Electrode(data_L)  # 左脑输出特征
        feature_R = ELE_Net_Electrode(data_R)  # 右脑输出特征
        feature = torch.cat([feature_L, feature_R], 2)  # 拼接左右脑数据 (280, 16, 10, 326)

    feature_L = feature_L.cpu().detach().numpy()  # 转换为 ndarray
    trial, filter, channel, sample = feature_L.shape  # 读取各维度大小
    feature_r1 = feature_L[:, :, 0:15, :]  # 区域1 电极特征 [280, 8, 15, 326]
    feature_ele_r1 = (np.transpose(feature_r1, [2, 0, 1, 3]).
                        reshape(channel_region, trial * filter * sample))  # 合并特征 [region_ch, feature]

    # pca = PCA(n_components=3)  # pca参数设定
    # feature_ele_r1 = pca.fit_transform(feature_trans_r1)  # 某区域的三维电极特征 [region_ch, 3]
    return feature_ele_r1

'''提取从ch_s到ch_e电极编号的电极特征'''
def feature_ele_r(subject, ELE_Net, ch_s, ch_e, channel_region):
    ELE_Net_Electrode = nn.Sequential(*list(ELE_Net.children()))[:1]
    test_data_L, test_data_R, test_label, te_num = read_test_data(Basic_folder, Current_datasets, subject)
    test_data = Data.TensorDataset(torch.from_numpy(test_data_L.astype(np.float32)),
                                   torch.from_numpy(test_data_R.astype(np.float32)),
                                   torch.from_numpy(test_label.astype(np.float32)))

    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    '''电极特征-aa'''

    for data_L, data_R, label in test_loader:
        data_L, data_R, label = data_L.cuda(), data_R.cuda(), label.cuda()
        data_L, data_R, label = Variable(data_L), Variable(data_R), Variable(label)
        feature_L = ELE_Net_Electrode(data_L)  # 左脑输出特征
        feature_R = ELE_Net_Electrode(data_R)  # 右脑输出特征
        feature = torch.cat([feature_L, feature_R], 2)  # 拼接左右脑数据 (280, 16, 10, 326)

    feature_L = feature_L.cpu().detach().numpy()  # 转换为 ndarray
    trial, filter, channel, sample = feature_L.shape  # 读取各维度大小
    feature_r = feature_L[:, :, ch_s:ch_e, :]  # 区域1 电极特征 [280, 8, 15, 326]
    feature_ele_r = (np.transpose(feature_r, [2, 0, 1, 3]).
                        reshape(channel_region, trial * filter * sample))  # 合并特征 [region_ch, feature]

    # pca = PCA(n_components=3)  # pca参数设定
    # feature_ele_r1 = pca.fit_transform(feature_trans_r1)  # 某区域的三维电极特征 [region_ch, 3]
    return feature_ele_r


'''返回左脑全部的电极特征'''
def feature_ele_L(subject, ELE_Net, channel_region):
    ch_s, ch_e = 0, 15
    feature_ele = feature_ele_r(subject, ELE_Net, ch_s, ch_e, channel_region)
    for re in range(1, 5):
        ch_s = re * 10
        ch_e = ch_s + 15
        _feature_ele = feature_ele_r(subject, ELE_Net, ch_s, ch_e, channel_region)
        feature_ele = np.append(feature_ele, _feature_ele, axis=0)
    return feature_ele


'''参数设定'''
# 读取数据与参数设定
Basic_folder = 'Datasets_Transfer_Task/BCI III IVa/'  # 服务器端根目录
Current_datasets = 'AllPermutatedData/'  # 数据集路径
subject_list = ['aa', 'al', 'av', 'aw', 'ay']  # 用户列表t
Cross_Mission = 0
batch_size = 280
channel_region = 15

'''读取左脑所有区域特征'''
ELE_Net = torch.load("Model/EX1/Task-1/Cross-1_epoch-149.pth")  # 加载模型
subject = subject_list[0]
feature_ele_EX1_aa = feature_ele_L(subject, ELE_Net, channel_region)
subject = subject_list[1]
feature_ele_EX1_al = feature_ele_L(subject, ELE_Net, channel_region)

ELE_Net = torch.load("Model/EX5/Task-1/Cross-1_epoch-149.pth")  # 加载模型
subject = subject_list[0]
feature_ele_EX5_aa = feature_ele_L(subject, ELE_Net, channel_region)
subject = subject_list[1]
feature_ele_EX5_al = feature_ele_L(subject, ELE_Net, channel_region)

feature_ele_EX1 = np.append(feature_ele_EX1_aa, feature_ele_EX1_al, axis=0)
feature_ele_EX5 = np.append(feature_ele_EX5_aa, feature_ele_EX5_al, axis=0)
feature_ele_EX1_5 = np.append(feature_ele_EX1, feature_ele_EX5, axis=0)

# pca = PCA(n_components=3)
# pca_feature_ele_EX1_5 = pca.fit_transform(feature_ele_EX1_5)
# np.save('PCAFeatureElectrode/pca_feature_ele_EX1_5.npy',pca_feature_ele_EX1_5)

tsne = TSNE(n_components=3, init='pca', random_state=500)
tsne_feature_ele_EX1_5 = tsne.fit_transform(feature_ele_EX1_5)
np.save('PCAFeatureElectrode/tsne_feature_ele_EX1_5.npy',tsne_feature_ele_EX1_5)


'''读取区域1 电极特征'''
ELE_Net = torch.load("Model/EX5/Task-1/Cross-1_epoch-149.pth")  # 加载模型
subject = subject_list[0]
feature_ele_r1_EX5_aa = feature_ele_r1(subject, ELE_Net, channel_region)
subject = subject_list[1]
feature_ele_r1_EX5_al = feature_ele_r1(subject, ELE_Net, channel_region)

ELE_Net = torch.load("Model/EX1/Task-1/Cross-1_epoch-149.pth")  # 加载模型
subject = subject_list[0]
feature_ele_r1_EX1_aa = feature_ele_r1(subject, ELE_Net, channel_region)
subject = subject_list[1]
feature_ele_r1_EX1_al = feature_ele_r1(subject, ELE_Net, channel_region)

feature_ele_r1_EX1 = np.append(feature_ele_r1_EX1_aa, feature_ele_r1_EX1_al, axis=0)
feature_ele_r1_EX5 = np.append(feature_ele_r1_EX5_aa, feature_ele_r1_EX5_al, axis=0)
feature_ele_r1_EX1_5 = np.append(feature_ele_r1_EX1, feature_ele_r1_EX5, axis=0)

# 进行各种基底的pca
pca = PCA(n_components=3)
pca_ele_r1_EX1_aa = pca.fit_transform(feature_ele_r1_EX1_aa)
pca_ele_r1_EX1_al = pca.fit_transform(feature_ele_r1_EX1_al)
pca_ele_r1_EX5_aa = pca.fit_transform(feature_ele_r1_EX5_aa)
pca_ele_r1_EX5_al = pca.fit_transform(feature_ele_r1_EX5_al)
pca_ele_r1_EX1 = pca.fit_transform(feature_ele_r1_EX1)
pca_ele_r1_EX5 = pca.fit_transform(feature_ele_r1_EX5)
pca_ele_r1_EX1_5 = pca.fit_transform(feature_ele_r1_EX1_5)

# 储存npy
np.save('PCAFeatureElectrode/pca_ele_r1_EX1_aa.npy', pca_ele_r1_EX1_aa)
np.save('PCAFeatureElectrode/pca_ele_r1_EX1_al.npy', pca_ele_r1_EX1_al)
np.save('PCAFeatureElectrode/feature_ele_r1_EX5_aa.npy', feature_ele_r1_EX5_aa)
np.save('PCAFeatureElectrode/feature_ele_r1_EX5_al.npy', feature_ele_r1_EX5_al)
np.save('PCAFeatureElectrode/feature_ele_r1_EX1.npy', feature_ele_r1_EX1)
np.save('PCAFeatureElectrode/feature_ele_r1_EX5.npy', feature_ele_r1_EX5)
np.save('PCAFeatureElectrode/feature_ele_r1_EX1_5.npy', feature_ele_r1_EX1_5)


