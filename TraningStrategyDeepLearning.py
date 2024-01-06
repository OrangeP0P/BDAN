global EX
from scipy.special import comb, perm

def tranning_strategy(EX):
    if EX == 1:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 2:  # no loss: EEGInception
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGInception'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2)) # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 3:  # no loss: DeepConvNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'DeepConvNet'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2)) # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 4:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGShallowConvNet'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 5:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet_1'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 6:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet_2'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 7:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet_3'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    elif EX == 8:  # no loss: EEGNet
        Current_Datasets = 'BCI IV 2a/'  # 数据集路径：原始数据集经过FIR滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet_4'  # 选取网络类型：PSCNN 数据不进行重排，只区分左右脑
        subject_list = ['1', '2', '3', '4', '5']  # 用户列表
        total_task = int(perm(len(subject_list), 2))  # 总任务数
        data_size = 280  # 数据集样本数

    return Current_Datasets, Net_number, subject_list, total_task, data_size
