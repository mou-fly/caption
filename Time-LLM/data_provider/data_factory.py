import os

from sympy import false

from data_provider.data_loader import Dataset_IMU, Dataset_ETT_hour
from torch.utils.data import DataLoader

data_dict = {
    'imu01': Dataset_IMU,
    'ETTh1':Dataset_ETT_hour
}


def data_provider(args, flag, path):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        # label_dict=args.label_dict,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns,
        # path = path
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader


def get_session(args, flag):
    dict_path = "C:\\project\\Time-LLM\\dataset\\ClassifiedTestTrain\\ClassifiedTestTrain\\{}".format(flag)
    file_list = os.listdir(dict_path)
    file_list = [f for f in file_list if os.path.isfile(os.path.join(dict_path, f))]
    data_list = []
    loader_list = []
    for f in file_list:
        path = os.path.join(dict_path, f)
        test_data, test_loader = data_provider(args, 'test', path)
        data_list.append(test_data)
        loader_list.append(test_loader)
        data_list.append(test_data)

    dict_path1 = "C:\\project\\Time-LLM\\dataset\\ClassifiedTestTrain\\ClassifiedTestTrain\\test\\session1.csv"
    # dict_path2 = "C:\\project\\Time-LLM\\dataset\\ClassifiedTestTrain\\ClassifiedTestTrain\\test\\session3.csv"
    # dict_path3 = "C:\\project\\Time-LLM\\dataset\\ClassifiedTestTrain\\ClassifiedTestTrain\\test\\session4.csv"
    data_list = []
    loader_list = []
    test_data, test_loader = data_provider(args, 'test', dict_path1)
    data_list.append(test_data)
    loader_list.append(test_loader)
    # data_list.append(test_data)
    # test_data, test_loader = data_provider(args, 'test', dict_path2)
    # data_list.append(test_data)
    # loader_list.append(test_loader)
    # data_list.append(test_data)
    # test_data, test_loader = data_provider(args, 'test', dict_path3)
    # data_list.append(test_data)
    # loader_list.append(test_loader)
    # data_list.append(test_data)

    return data_list, loader_list

