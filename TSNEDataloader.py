import numpy as np
import scipy.io as scio


def read_test_data(Basic_folder, Current_Datasets, subject_2):
    te_num = 280  # 数据集样本大小
    # 数据集指定

    tdn = 'data_L_' + subject_2
    tdp = Basic_folder + Current_Datasets + tdn + '.mat'
    tln = 'label_' + subject_2
    tlp = Basic_folder + Current_Datasets + tln + '.mat'

    target_data = scio.loadmat(tdp)[tdn]
    target_label = scio.loadmat(tlp)[tln]

    test_target_data = target_data[:, :]

    _test_target_label = target_label[:, :]
    test_target_label = np.array(range(0, len(_test_target_label)))

    for i in range(0, len(_test_target_label)):
        test_target_label[i] = _test_target_label[i]

    test_target_label = test_target_label - 1

    test_target_data = np.transpose(np.expand_dims(test_target_data, axis=1), (0, 1, 3, 2))


    mask_L = np.where(np.isnan(test_target_data))
    test_target_data[mask_L] = np.nanmean(test_target_data)

    return test_target_data, test_target_label, te_num
