import os
import time

import numpy as np
import torch.nn.functional as F
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import optim, nn
from tqdm import tqdm
# from utils.rcg_database import get_rcg_database
# from transformers.models.swiftformer.convert_swiftformer_original_to_hf import device

from modules.simple_tokenizer import SimpleTokenizer
from data_provider.data_factory import data_provider
from utils.cal_loss import score_metric, tokenizer, metric

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def train(configs, model, model_configs):
    train_data, train_loader = data_provider(configs, 'train')
    vaild_data, valid_loader = data_provider(configs, 'val')
    test_data, test_loader = data_provider(configs, 'test')

    if configs.model == 'clip4caption':
        if configs.clip_pretrain:
            model.pretrain(train_loader, valid_loader, configs)
        preModel = model.load_pretrain()

    if configs.model == 'rcg':
        if configs.rcg_get_database:
            get_rcg_database(configs)

    if configs.model == 'vtar':
        database_x = []
        database_y = []
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(valid_loader)):
            for j in range(len(batch_x)):
                x = batch_x[j].view(-1).to(configs.device)
                y = batch_y[j].to(configs.device)
                database_x.append(x)
                database_y.append(y)
        database_x = torch.stack(database_x)

    model_optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    train_steps = len(train_loader)

    check_path = "/home/wangtiantian/dengfei/caption/checkpoint_30s"
    res_path = "/home/wangtiantian/dengfei/caption/result_30s"
    check_path = os.path.join(check_path, model_configs)

    # train_loader, vali_loader, test_loader, model, model_optimizer = accelerator.prepare(
    #     train_loader, valid_loader, test_loader, model, model_optimizer)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = np.inf
    best_bleu4 = 0
    cnt = 0
    patience = configs.patience
    delta = configs.delta
    all_time = time.time()
    for epoch in range(configs.epoch):
        iter_count = 0
        train_loss = []
        model.train()
        epoch_time = time.time()
        # test_score = test(configs, model, test_loader)
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(train_loader)):
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=configs.device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=configs.device)  # 16, 30
            iter_count += 1
            model_optimizer.zero_grad()
            if configs.model == 'transformer' or configs.model == 'rcg' or configs.model == 'diary' or configs.model == 'model':
                output, itc_loss = model(batch_x, batch_y[:, :-1])
            elif configs.model == 'vtar':
                output = model(batch_x, batch_y[:, :-1], database_x, database_y)
            elif configs.model == 'clip4caption':
                output = model(batch_x, batch_y[:, :-1], preModel)
            else:
                output = model(batch_x, batch_y[:, :-1])  # 8, 30, 556
            pred = output.view(-1, 556)
            truth = batch_y.view(-1)
            if (configs.model == 'transformer' or configs.model == 'lstm' or configs.model == 'vtar'
                    or configs.model == 'clip4caption' or configs.model == 'rcg'
                    or configs.model == 'diary' or configs.model == 'vtar'
                    or configs.model == 'al' or configs.model == 'model'
                    or configs.model == 'xrf' or configs.model == 'recap'
            ):
                truth = batch_y[:, 1:].reshape(-1)
            ce_loss = criterion(pred, truth)
            # loss = ce_loss + itc_loss
            loss = ce_loss
            loss.backward()
            model_optimizer.step()
            train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                iter_count = 0
        if configs.model == 'clip4caption':
            valid_loss = valid(configs, model, valid_loader, criterion, None, None, preModel)
        elif configs.model == 'vtar':
            valid_loss = valid(configs, model, valid_loader, criterion, database_x, database_y, None)
        elif (configs.model == 'transformer' or configs.model == 'rcg' or configs.model == 'diary' or configs.model == 'al' 
                        or configs.model == 'model' or configs.model == 'xrf' or configs.model == 'recap' 
            ):
            valid_loss = valid(configs, model, valid_loader, criterion)

        # test_score = test(configs, model, test_loader)
        # new_bleu4 = test_score[3]
        # if epoch % 5 == 0:
        res_name = model_configs + ".txt"
        res_file = os.path.join(res_path, res_name)
        res_epoch = ("Epoch: {}   epoch time: {}  all time: {} \n".format(epoch + 1, time.time() - epoch_time, time.time() - all_time))
        # res = ("Bleu-1:{0}, Bleu-2:{1}, Bleu-3:{2}, Bleu-4:{3}, Meteor:{4}, Rouge-L:{5}, CIDEr:{6}\n".format(
        #     test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], test_score[5], test_score[6]))
        res_loss = ("train_loss:{0}, valid_loss:{1}\n".format(np.mean(train_loss), valid_loss))
        with open(res_file, "a", encoding="utf-8") as file:
            file.write(res_epoch)
            file.write(res_loss)
            # file.write(res)
        os.makedirs(check_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(check_path, "all.pth"))
        if valid_loss + delta < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(check_path, "best_loss.pth"))
            print("save better loss...")
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                print("early stop")
                break
        # if new_bleu4 > best_bleu4:
        #     best_bleu4 = new_bleu4
        #     torch.save(model.state_dict(), os.path.join(check_path, "best_bleu.pth"))
        #     print("save better bleu...")
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        print(model_configs)
        print("Epoch: {0}, Train Loss: {1:.7f}".format(epoch + 1, np.mean(train_loss)))
        print("Epoch: {0}, Valid Loss: {1:.7f}".format(epoch + 1, valid_loss))
        # print("Bleu-1:{0}, Bleu-2:{1}, Bleu-3:{2}, Bleu-4:{3}, Meteor:{4}, Rouge-L:{5}, CIDEr:{6}".format(
        #     test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], test_score[5], test_score[6]))
        print("patience : {} / {}".format(cnt, patience))

def test(configs, model, test_loader):
    model.eval()
    total_loss = []
    tokenizer = SimpleTokenizer()
    max_len = configs.caption_max_len
    with torch.no_grad():
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(test_loader)):
            if i == 2:
                break
            index = i
            lens = (mask == 1).sum(dim=1)
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=configs.device)
            if configs.model == 'transformer':
                preds = []
                for ii in range(configs.batch_size):
                    outputs = [1]
                    src = batch_x[ii].unsqueeze(0)
                    for j in range(max_len):
                        caption = torch.tensor(outputs, dtype=torch.long, device=configs.device).unsqueeze(0)
                        with torch.no_grad():
                            output, _ = model(src, caption)
                        best_guess = output.argmax(2)[:, -1].item()
                        outputs.append(best_guess)
                        # if best_guess == 2:
                        #     break
                    preds.append(outputs)
            else:
                batch_y = torch.tensor(batch_y, dtype=torch.long, device=configs.device)
                output = model(batch_x, batch_y).long()
                probs = F.softmax(output.float(), dim=-1)
                preds = torch.argmax(probs, dim=-1)
            score = score_metric(preds, batch_y, lens, index).cpu()
            total_loss.append(score)

        total_loss = np.array(total_loss)
        total_loss = np.mean(total_loss, axis=0)

    return total_loss

def valid(configs, model, valid_loader, criterion, database_x=None, database_y=None, preModel=None):
    model.eval()
    total_loss = []
    with (torch.no_grad()):
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(valid_loader)):
            lens = (mask == 1).sum(dim=1)
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=configs.device)
            batch_y = torch.tensor(batch_y, dtype=torch.long, device=configs.device)
            if configs.model == 'transformer' or configs.model == 'rcg' or configs.model == 'model':
                output, itc_loss = model(batch_x, batch_y[:, :-1])
            elif configs.model == 'vtar':
                output = model(batch_x, batch_y[:, :-1], database_x, database_y)
            elif configs.model == 'clip4caption':
                output = model(batch_x, batch_y[:, :-1], preModel)
            else:
                output = model(batch_x, batch_y[:, :-1])  # 8, 30, 556
            pred = output.view(-1, 556)
            truth = batch_y.view(-1)
            if (configs.model == 'transformer' or configs.model == 'lstm' or configs.model == 'vtar'
                or configs.model == 'clip4caption' or configs.model == 'rcg' or configs.model == 'lstm'
                or configs.model == 'al' or configs.model == 'model' or configs.model == 'xrf' or configs.model == 'recap'
            ):
                truth = batch_y[:, 1:].reshape(-1)
            loss = criterion(pred, truth)
            total_loss.append(loss)
        average = sum(total_loss) / len(total_loss)

    return average

def ar_infer(configs, model, sequence):  # seq [5, 1500, 6]
    # model.load_state_dict(torch.load("model.pth"))
    model.eval()  # 如果是推理/验证阶段
    device = configs.device
    max_len = configs.caption_max_len
    src = sequence.unsqueeze(0)  # seq[1, 5, 1500, 6]
    src = torch.tensor(src, dtype=torch.float32, device=device)
    outputs = [1]
    for i in range(max_len - 1):
        caption = torch.tensor(outputs).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model(src, caption)
        best_guess = output.argmax(2)[:, -1].item()
        outputs.append(best_guess)
        if best_guess == 2:
            break
    text = tokenizer.decode(outputs)
    return text

def infer(configs, model, model_name, f=None):
    test_data, test_loader = data_provider(configs, 'test')
    total_loss = []
    total_rmc = []
    f = f
    model.eval()
    infer_path = "/home/wangtiantian/dengfei/caption/checkpoint_30s/transformer_1_patch:False_False_300_150_encoder:2_frames:1500_JE:False_PE:t_ITC:False_conv:False_pods/best_loss.pth"
    if configs.model == 'clip4caption':
        preModel = model.load_pretrain()
    if configs.model == 'vtar':
        vaild_data, valid_loader = data_provider(configs, 'val')
        database_x = []
        database_y = []
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(valid_loader)):
            for j in range(len(batch_x)):
                x = batch_x[j].view(-1).to(configs.device)
                y = batch_y[j].to(configs.device)
                database_x.append(x)
                database_y.append(y)
        database_x = torch.stack(database_x)
    state_dict = torch.load(infer_path, map_location='cuda:0')
    model.load_state_dict(state_dict)
    model.to(configs.device)
    # model.load_state_dict(torch.load(infer_path),map_location='cuda:0')
    # model.to(configs.device)
    max_len = configs.caption_max_len
    tokenizer = SimpleTokenizer()
    with torch.no_grad():
        for i, (batch_x, batch_y, mask) in tqdm(enumerate(test_loader)):
            index = 1
            lens = (mask == 1).sum(dim=1)
            batch_x = torch.tensor(batch_x, dtype=torch.float32, device=configs.device)
            if configs.model == 'transformer' or configs.model == 'vtar' or configs.model == 'clip4caption' or configs.model == 'xrf' or configs.model == 'recap':
                preds = []
                for ii in range(configs.batch_size):
                    outputs = [1]
                    src = batch_x[ii].unsqueeze(0)
                    for j in range(max_len):
                        caption = torch.tensor(outputs, dtype=torch.long, device=configs.device).unsqueeze(0)
                        if configs.model == 'transformer':
                            output, _ = model(src, caption)
                        elif configs.model == 'vtar':
                            output = model(src, caption, database_x, database_y)
                        elif configs.model == 'clip4caption':
                            output = model(src, caption, preModel)
                        else: 
                            output = model(src, caption)
                        best_guess = output.argmax(2)[:, -1].item()
                        outputs.append(best_guess)
                    preds.append(outputs)
            elif configs.model == 'lstm' or configs.model == 'al':
                preds = []
                captions = model(batch_x, batch_y, False)
                caption = torch.stack(captions, dim=0)
                caption = caption.transpose(0, 1).cpu()
                for c in caption:
                    preds.append(c)
            if configs.expand_time != 30:
                score, rmc = metric(preds, batch_y, lens, configs.expand_time)
            else:
                score = score_metric(preds, batch_y, lens, index, f).cpu()
            total_loss.append(score)
            if configs.expand_time != 30:
                total_rmc.append(rmc)

        total_loss = np.array(total_loss)
        test_score = np.mean(total_loss, axis=0)
        total_rmc = np.array(total_rmc)
        rmc = np.mean(total_rmc, axis=0)

    score = ("Bleu-1:{0}, Bleu-2:{1}, Bleu-3:{2}, Bleu-4:{3}, Meteor:{4}, Rouge-L:{5}, CIDEr:{6}".format(
        test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], test_score[5], test_score[6]))
    if configs.expand_time != 30:
        score = ("Bleu-1:{0}, Bleu-2:{1}, Bleu-3:{2}, Bleu-4:{3}, Meteor:{4}, Rouge-L:{5}, CIDEr:{6}, RMC:{7}".format(
            test_score[0], test_score[1], test_score[2], test_score[3], test_score[4], test_score[5], test_score[6], rmc))
    print(score)
    infer_folder = "/home/wangtiantian/dengfei/caption/infer_30s"
    infer_folder = os.path.join(infer_folder, model_name) if configs.expand_time == 30 else os.path.join(infer_folder, configs.expand_time)
    # infer_folder = os.path.join(infer_folder, model_name) if configs.device_index == 0 else os.path.join(infer_folder, configs.device_index)
    infer_file = infer_folder + ".txt"
    with open(infer_file, 'a') as f:
        f.write(score)
        f.write("\n")

