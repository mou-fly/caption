import os

from data_provider.data_loader import Dataset_raw, Dataset_img
from torch.utils.data import DataLoader

data_dict = {
    'raw': Dataset_raw,
    'img': Dataset_img
}

def data_provider(args, flag):
    Data = data_dict[args.modality]
    if flag == 'test':
        data_path = args.test_data_path
        label_path = args.test_label_path
    else:
        data_path = args.train_data_path
        label_path = args.train_label_path

    data_set = Data(
        root_path = args.root_path,
        data_path = data_path,
        flag = flag,
        use_patch = args.use_patch,
        label_path = label_path,
        caption_len = args.caption_max_len,
        patch_len = args.patch_len,
        stride = args.stride
    )

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )

    return data_set, data_loader
