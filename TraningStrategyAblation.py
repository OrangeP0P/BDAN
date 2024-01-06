global EX
from scipy.special import comb, perm

def tranning_strategy(EX):

    if EX == 1:  # no loss: PSCNN
        l_s = 1  # la_1
        l_t = 1
        l_cls = 1
        l_s_start = 0
        l_t_start = 50
        loss_speed_s = 50
        loss_speed_t = 50
        subject_list = ['aa', 'al', 'av', 'ay', 'aw']  # 用户列表
        total_task = int(perm(5, 2))  # 总任务数
        data_size = 280  # 数据集样本数
        Current_Datasets = 'BCI III IVa/'  # 数据集路径：原始数据集经过Fir滤波，不进行重排，不剔除参考电极
        Net_number = 'TEGDAN_1'  # 选取网络类型：PSCNN 数据进行重排，区分左右脑

    if EX == 2:  # no loss: PSCNN
        l_s = 1  # la_1
        l_t = 1
        l_cls = 1
        l_s_start = 0
        l_t_start = 50
        loss_speed_s = 50
        loss_speed_t = 50
        subject_list = ['aa', 'al', 'av', 'ay', 'aw']  # 用户列表
        total_task = int(perm(5, 2))  # 总任务数
        data_size = 280  # 数据集样本数
        Current_Datasets = 'BCI III IVa/'  # 数据集路径：原始数据集经过Fir滤波，不进行重排，不剔除参考电极
        Net_number = 'TEGDAN_2'  # 选取网络类型：PSCNN 数据进行重排，区分左右脑

    if EX == 3:  # no loss: PSCNN
        l_s = 0  # la_1
        l_t = 0
        l_cls = 1
        l_s_start = 0
        l_t_start = 50
        loss_speed_s = 50
        loss_speed_t = 50
        subject_list = ['aa', 'al', 'av', 'ay', 'aw']  # 用户列表
        total_task = int(perm(5, 2))  # 总任务数
        data_size = 280  # 数据集样本数
        Current_Datasets = 'BCI III IVa/'  # 数据集路径：原始数据集经过Fir滤波，不进行重排，不剔除参考电极
        Net_number = 'EEGNet'  # 选取网络类型：PSCNN 数据进行重排，区分左右脑

    if EX == 4:  # no loss: PSCNN
        l_s = 1  # la_1
        l_t = 1
        l_cls = 1
        l_s_start = 0
        l_t_start = 50
        loss_speed_s = 50
        loss_speed_t = 50
        subject_list = ['aa', 'al', 'av', 'ay', 'aw']  # 用户列表
        total_task = int(perm(5, 2))  # 总任务数
        data_size = 280  # 数据集样本数
        Current_Datasets = 'BCI III IVa/'  # 数据集路径：原始数据集经过Fir滤波，不进行重排，不剔除参考电极
        Net_number = 'TEGDAN_4'  # 选取网络类型：PSCNN 数据进行重排，区分左右脑

    if EX == 5:  # no loss: PSCNN
        l_s = 1  # la_1
        l_t = 1
        l_cls = 1
        l_s_start = 0
        l_t_start = 50
        loss_speed_s = 50
        loss_speed_t = 50
        subject_list = ['aa', 'al', 'av', 'ay', 'aw']  # 用户列表
        total_task = int(perm(5, 2))  # 总任务数
        data_size = 280  # 数据集样本数
        Current_Datasets = 'BCI III IVa/'  # 数据集路径：原始数据集经过Fir滤波，不进行重排，不剔除参考电极
        Net_number = 'TEGDAN_5'  # 选取网络类型：PSCNN 数据进行重排，区分左右脑

    return l_s, l_t, l_cls, l_s_start, l_t_start, loss_speed_s, loss_speed_t,\
           Current_Datasets, Net_number, subject_list, total_task, data_size

