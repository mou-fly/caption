import argparse
import random
from datetime import time
import time

from PIL.ImageOps import expand
# from accelerate import Accelerator, DeepSpeedPlugin, optimizer
from accelerate import DistributedDataParallelKwargs
import numpy as np
import torch
from sympy import false


from exp.exp import train, valid, test, infer
from models import (lstm, attention_lstm, transformer, clip4caption, encoder_only, model, vtar, rcg,
                    rf_diary, nacf, al, xrfmamba, vidrecap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IMU-LLM')

    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='caption')
    parser.add_argument("--model", type=str, required=True, default='mamba')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2, help='batch size of train input data')
    parser.add_argument('--train_data_path', type=str, default="data_train.h5")
    parser.add_argument('--test_data_path', type=str, default="data_test.h5")
    parser.add_argument('--train_label_path', type=str, default="label_train.csv")
    parser.add_argument('--test_label_path', type=str, default="label_test.csv")
    parser.add_argument("--mode", type=int, required=True, default=0, help="0-train, 1-test, 2-use")
    parser.add_argument("--modality", type=str, required=True, default='raw')
    parser.add_argument("--raw_dim", type=int, default=36)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--caption_max_len", type=int, default=45)  #
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--use_patch", type=bool, default=False)     #
    parser.add_argument("--patch_len", type=int, default=20)       #
    parser.add_argument("--stride", type=int, default=5)          #
    parser.add_argument("--n_frames", type=int, default=1500)       #
    parser.add_argument("--joints_emb", type=bool, default=False)    #
    parser.add_argument("--itc_loss", type=bool, default=False)     #
    parser.add_argument("--itc_loss_tatio", type=int, default=0.1)
    parser.add_argument("--conv_ffn", type=bool, default=False)      #
    parser.add_argument("--expand_time", type=int, default=90)


    # transformer config
    parser.add_argument("--transformer_d_model", type=int, required=False, default=512)
    parser.add_argument("--transformer_n_heads", type=int, required=False, default=8)
    parser.add_argument("--transformer_forward_expansion", type=int, required=False, default=4)
    parser.add_argument("--transformer_encoder_n_layers", type=int, required=False, default=2)          # 
    parser.add_argument("--transformer_decoder_n_layers", type=int, required=False, default=3)
    parser.add_argument("--transformer_n_layers", type=int, required=False, default=1)
    parser.add_argument("--transformer_patch", type=bool, required=False, default=False)                # 
    parser.add_argument("--transformer_PE", type=str, required=False, default="t")

    # model config
    parser.add_argument("--model_d_model", type=int, required=False, default=512)
    parser.add_argument("--model_n_heads", type=int, required=False, default=8)
    parser.add_argument("--model_forward_expansion", type=int, required=False, default=4)
    parser.add_argument("--model_encoder_n_layers", type=int, required=False, default=2)  #
    parser.add_argument("--model_fusion_n_layers", type=int, required=False, default=2)  #            #

    # S2VT
    parser.add_argument("--s2vt_input_dim", type=int, required=False, default=96)
    parser.add_argument("--s2vt_drop", type=int, required=False, default=0.3)

    # vtar
    parser.add_argument("--vtar_encoder_layers", type=int, required=False, default=2)
    parser.add_argument("--vtar_d_model", type=int, required=False, default=512)
    parser.add_argument("--vtar_n_heads", type=int, required=False, default=8)
    parser.add_argument("--vtar_forward_expansion", type=int, required=False, default=4)

    # clip4caption
    parser.add_argument("--clip_pretrain", type=bool, required=False, default=False)
    parser.add_argument("--clip_forward_expansion", type=int, required=False, default=4)
    parser.add_argument("--clip_d_model", type=int, required=False, default=512)
    parser.add_argument("--clip_n_heads", type=int, required=False, default=8)

    # rcg
    parser.add_argument("--rcg_d", type=int, required=False, default=512)
    parser.add_argument("--rcg_n_heads", type=int, required=False, default=8)
    parser.add_argument("--rcg_model", type=int, required=False, default=512)
    parser.add_argument("--rcg_encoder_layers", type=int, required=False, default=1)
    parser.add_argument("--rcg_forward_expansion", type=int, required=False, default=4)
    parser.add_argument("--rcg_get_database", type=bool, required=False, default=False)

    # diary
    parser.add_argument("--diary_d", type=int, required=False, default=512)

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--patience', type=int, default=10)   #
    parser.add_argument('--delta', type=float, default=0.01)  #
    parser.add_argument("--device_index", type=int, default=28)

    args = parser.parse_args()
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


    if args.model == 'clip4caption':
        model = clip4caption.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'vtar':
        model = vtar.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'lstm':
        model = lstm.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'transformer':
        model = transformer.Model(args)
        model_configs = ("{}_{}_patch:{}_{}_{}_{}_encoder:{}_frames:{}_JE:{}_PE:{}_ITC:{}_conv:{}".format(
            args.model, args.transformer_n_layers, args.use_patch, args.transformer_patch, args.patch_len, args.stride, 
            args.transformer_encoder_n_layers, args.n_frames, args.joints_emb, args.transformer_PE, args.itc_loss, args.conv_ffn))
    elif args.model == 'encoder':
        model = encoder_only.Model(args)
        model_configs = ("{0}_{1}_patch:{2}".format(args.model, args.transformer_n_layers, args.use_patch))
    elif args.model == 'rcg':
        model = rcg.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'al':
        model = al.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'nacf':
        model = nacf.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'diary':
        model = rf_diary.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'xrf':
        model = xrfmamba.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == 'recap':
        model = vidrecap.Model(args)
        model_configs = ("{}".format(args.model))
    elif args.model == "model":
        model = model.Model(args)
        model_configs = ("{}_patch:{}_{}_{}_encoder:{}_fusion:{}_frames:{}_JE:{}_conv:{}".format(
            args.model, args.use_patch, args.patch_len, args.stride,
            args.model_encoder_n_layers, args.model_fusion_n_layers, args.n_frames, args.joints_emb, args.conv_ffn))

    model = model.to(args.device).float()



    if args.mode == 0:
        setting = '{}_{}'.format(
            args.task_name,
            args.model,
        )
        print(">>>>>>>>>>>train<<<<<<<<<<<<")
        train(args, model, model_configs)
    else:
        infer(args, model, model_configs, True)
