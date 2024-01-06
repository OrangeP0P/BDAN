# #
# # # data_size = 280  # 数据集样本数(包含训练集和验证集)
# # # split_num = 10  # 将模型集分割的份数
# # # train_source_list, train_target_list, validation_target_list = split_data(split_num, data_size)  # 读取训练集、验证集、测试集的序号
# # # np.save('./train_source_list.npy', train_source_list)  # 保存交叉验证序号矩阵：训练集
# # # np.save('./train_target_list.npy', train_target_list)  # 保存交叉验证序号矩阵：训练集
# # # np.save('./validation_target_list.npy', validation_target_list)  # 保存交叉验证序号矩阵：测试集
# #
#

# channel_location = pd.read_excel('region_location.xls')
# Xe = channel_location.X
# Ye = channel_location.Y
# RT = np.ones([len(Xe), len(Ye)])
# for i in range(0, len(Xe)):
#     for j in range(0, len(Ye)):
#         RT[i, j] = math.sqrt( (Xe[i]-Xe[j])*(Xe[i]-Xe[j]) + (Ye[i]-Ye[j])*(Ye[i]-Ye[j]))
#         if RT[i, j] != 0:
#             RT[i, j] =  1/(RT[i, j]*RT[i, j])
# np.save('./RT.npy', RT)