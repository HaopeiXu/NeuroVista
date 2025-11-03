"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""

import os
import argparse
import random
import itertools
import datetime
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange


gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
result_path = '/mnt/bn/haopei-personal-v1/work/NICE-EEG-main/' 
model_idx = 'test0'
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='100', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=3000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--ratio', default=0.1, type=float,
                    help='mask_ratio.')



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# def weights_init_normal(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='conv2d')
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.ones_(m.weight)
#         nn.init.zeros_(m.bias)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 


# Image2EEG
class IE():
    def __init__(self, args, train_subs, test_sub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
        self.train_subs = train_subs
        self.test_sub = test_sub
        self.batch_size = args.batch_size
        self.batch_size_test = 400
        self.batch_size_img = 500 
        self.n_epochs = args.epoch

        self.lambda_cen = 0.003
        self.alpha = 0.5

        self.proj_dim = 256

        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        # self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/mnt/bn/haopei-personal-v1/work/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = './dnn_feature/'
        self.test_center_path = './dnn_feature/'
        self.pretrain = False

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')

    def get_eeg_train_data(self):
        args = parser.parse_args()
        
        all_train = []
        for sub in self.train_subs:
            file = os.path.join(self.eeg_data_path, 'sub-' + format(sub, '02'), 'preprocessed_eeg_training.npy')
            data = np.load(file, allow_pickle=True)
            data = data['preprocessed_eeg_data']  # 原始形状 (16540, 4, 63, 250)

            # 参数
            n_samples, n_repeats, n_channels, n_times = data.shape
            mask_ratio = 0.2  # 要mask的比例
            n_mask = int(n_channels * mask_ratio)  # 要mask多少个通道

            # 随机 mask 每个样本
            for i in range(n_samples):
                mask_channels = np.random.choice(n_channels, n_mask, replace=False)
                data[i, :, mask_channels, :] = 0.0  # 或者用 np.nan 或其它方式掩盖

            data = np.mean(data, axis=1)            # 变为 (16540, 63, 250)
            data = np.expand_dims(data, axis=1)       # 变为 (16540, 1, 63, 250)
            all_train.append(data)
        train_data = np.concatenate(all_train, axis=0)
        return train_data

    # 加载测试受试者的EEG数据
    def get_eeg_test_data(self):
        file = os.path.join(self.eeg_data_path, 'sub-' + format(self.test_sub, '02'), 'preprocessed_eeg_test.npy')
        data = np.load(file, allow_pickle=True)
        data = data['preprocessed_eeg_data']  # (200, 80, 63, 250)
        data = np.mean(data, axis=1)           # (200, 63, 250)
        data = np.expand_dims(data, axis=1)      # (200, 1, 63, 250)
        test_label = np.arange(200)              # 测试标签：0~199
        return data, test_label

    # 加载训练受试者的图像特征（假设各受试者文件命名为：{dnn}_feature_maps_training_subXX.npy）
    def get_img_train_data(self):
        all_img = []
        for sub in self.train_subs:
            file = os.path.join(self.img_data_path, self.args.dnn + '_feature_maps_training' + '.npy')
            data = np.load(file, allow_pickle=True)
            data = np.squeeze(data)  # 假设形状为 (16540, 768) 每个受试者
            all_img.append(data)
        train_img_feature = np.concatenate(all_img, axis=0)
        return train_img_feature

    # 加载测试受试者的图像中心特征（假设文件命名为：center_{dnn}_subXX.npy）
    def get_img_test_data(self):
        file = os.path.join(self.test_center_path, 'center_' + self.args.dnn + '.npy')
        test_center = np.load(file, allow_pickle=True)
        test_center = np.squeeze(test_center)  # 形状 (200, 768)
        return test_center
    
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)

        # 加载训练数据：EEG和图像特征
        train_eeg = self.get_eeg_train_data()  # (N, 1, 63, 250)，N为所有训练样本数（9个受试者之和）
        train_img_feature = self.get_img_train_data()  # (N, 768)

        # 加载测试数据：仅测试受试者
        test_eeg, test_label = self.get_eeg_test_data()  # (200, 1, 63, 250) 和标签 (200,)
        test_center = self.get_img_test_data()           # (200, 768)


        train_eeg = torch.from_numpy(train_eeg)
        train_image = torch.from_numpy(train_img_feature)

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_eeg = torch.from_numpy(test_eeg)
        test_center = torch.from_numpy(test_center)
        test_label = torch.from_numpy(test_label)
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)


        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters()), lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()

            total_correct = 0
            total_samples = 0

            # starttime_epoch = datetime.datetime.now()

            for i, (eeg, img) in enumerate(self.dataloader):

                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                # label = Variable(label.cuda().type(self.LongTensor))
                labels = torch.arange(eeg.shape[0])  # used for the loss
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)
                # img_features = self.Enc_img(img).last_hidden_state[:,0,:]

                # project the features to a multimodal embedding space
                eeg_features = self.Proj_eeg(eeg_features)
                img_features = self.Proj_img(img_features)

                # normalize the features
                eeg_features = eeg_features / eeg_features.norm(dim=1, keepdim=True)
                img_features = img_features / img_features.norm(dim=1, keepdim=True)
                # print(img_features.shape)

                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t()  # (B, B)
                logits_per_img = logits_per_eeg.t()

                loss_eeg = self.criterion_cls(logits_per_eeg, labels)
                loss_img = self.criterion_cls(logits_per_img, labels)
                loss = (loss_eeg + loss_img) / 2

                preds_eeg = logits_per_eeg.argmax(dim=1)
                total_correct += (preds_eeg == labels).sum().item()
                total_samples += labels.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                overall_acc = total_correct / total_samples
            
                torch.save(self.Enc_eeg.module.state_dict(), './model-cross-mae/Enc_eeg_sub' + format(self.test_sub, '02') + '.pth')
                torch.save(self.Proj_eeg.module.state_dict(), './model-cross-mae/Proj_eeg_sub' + format(self.test_sub, '02') + '.pth')
                torch.save(self.Proj_img.module.state_dict(), './model-cross-mae/Proj_img_sub' + format(self.test_sub, '02') + '.pth')

                # print("Epoch {} train accuracy: {:.4f}%".format(e + 1, overall_acc*100))


        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model-cross-mae/Enc_eeg_sub' + format(self.test_sub, '02') + '.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model-cross-mae/Proj_eeg_sub' + format(self.test_sub, '02') + '.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model-cross-mae/Proj_img_sub' + format(self.test_sub, '02') + '.pth'), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))
                tlabel = Variable(tlabel.type(self.LongTensor))
                all_center = Variable(all_center.type(self.Tensor))            

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))
                tfea = tfea / tfea.norm(dim=1, keepdim=True)
                similarity = tfea @ all_center.t().softmax(dim=-1)
                _, indices = similarity.topk(5)

                tt_label = tlabel.view(-1, 1)
                total += tlabel.size(0)
                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('Test accuracy for subject {}: Top1-%.6f, Top3-%.6f, Top5-%.6f'.format(self.test_sub) % (top1_acc, top3_acc, top5_acc))
        return top1_acc, top3_acc, top5_acc


def main():
    args = parser.parse_args()
    num_sub = args.num_sub   # 总受试者数量，默认10
    all_top1 = []
    all_top3 = []
    all_top5 = []

    current_time = datetime.datetime.now()
    # 格式化时间为 "月日时分" 格式，例如 "03200945" 表示 3月20日09:45
    time_str = current_time.strftime("%m%d%H%M%S")

    # seed_n = np.random.randint(args.seed)
    seed_n = args.seed
    
    print('seed is ' + str(seed_n))
    print(f'ratio is {args.ratio}')

    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)

    for test_sub in range(1, num_sub+1):
        train_subs = [s for s in range(1, num_sub+1) if s != test_sub]
        print('Leave-one-out: Test subject {}'.format(test_sub))
        starttime = datetime.datetime.now()

        ie = IE(args, train_subs, test_sub)
        Acc, Acc3, Acc5 = ie.train()
        print('Subject {} Test Top1 Acc: {:.4f}'.format(test_sub, Acc))
        endtime = datetime.datetime.now()
        print('Subject {} duration: {}'.format(test_sub, endtime - starttime))
        all_top1.append(Acc)
        all_top3.append(Acc3)
        all_top5.append(Acc5)

    avg_top1 = np.mean(all_top1)
    avg_top3 = np.mean(all_top3)
    avg_top5 = np.mean(all_top5)
    print('Overall Test Accuracy:')
    print("all_top1",all_top1)
    print("all_top5",all_top5)
    print('Average Top1: {:.6f}, Top3: {:.6f}, Top5: {:.6f}'.format(avg_top1, avg_top3, avg_top5))


    results = {
        'Subject': list(range(1, num_sub+1)) + ['Average'],
        'Top1': all_top1 + [avg_top1],
        'Top3': all_top3 + [avg_top3],
        'Top5': all_top5 + [avg_top5],
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(result_path, f'result{seed_n}_{time_str}_mae_cross.csv'), index=False)
    print("Results saved.")


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))