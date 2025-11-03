"""
Object recognition Things-EEG2 dataset

use 250 Hz data
"""
import wandb
import os
os.environ["WANDB_MODE"] = "offline"

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

# import debugpy
# debugpy.listen(('0.0.0.0', 4000))  # 监听所有地址的5678端口
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # 等待VSCode的连接


gpus = [1]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
result_path = '/mnt/bn/haopei-personal-v1/work/NICE-EEG-main/results' 
 
parser = argparse.ArgumentParser(description='Experiment Stimuli Recognition test with CLIP encoder')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default='200', type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='number of subjects used in the experiments. ')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=2023, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--subject', default=1, type=int,
                    help='subject ID to train and test (from 1 to 10)')
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

# NICE
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


# TODO
# ------------------ (b) 双向跨模态注意力 ------------------ 
class CrossModalAttention(nn.Module):
    """
    双向注意力:
     - EEG->Image: 以 EEG 为 Query，Image 为 Key/Value
     - Image->EEG: 以 Image 为 Query，EEG 为 Key/Value
    注意: 这里假设输入形状都是 (B, seq_len, dim)，如果只有单向Token，可以让 seq_len=1
    """
    def __init__(self, embed_dim=768, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        
        self.mha_eeg_to_img = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.mha_img_to_eeg = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layernorm_eeg = nn.LayerNorm(embed_dim)
        self.layernorm_img = nn.LayerNorm(embed_dim)

    def forward(self, x_eeg, x_img):
        """
        x_eeg: (B, seq_eeg, embed_dim)
        x_img: (B, seq_img, embed_dim)
        return:
         out_eeg: (B, seq_eeg, embed_dim)
         out_img: (B, seq_img, embed_dim)
        """
        # ============ 1) EEG -> Image =============
        # 以 EEG 为 query, image 为 key & value
        attn_eeg2img, _ = self.mha_eeg_to_img(x_eeg, x_img, x_img)
        # 残差 + LN
        out_eeg = self.layernorm_eeg(x_eeg + self.dropout(attn_eeg2img))

        # ============ 2) Image -> EEG =============
        # 以 image 为 query, EEG 为 key & value
        attn_img2eeg, _ = self.mha_img_to_eeg(x_img, x_eeg, x_eeg)
        # 残差 + LN
        out_img = self.layernorm_img(x_img + self.dropout(attn_img2eeg))

        return out_eeg, out_img

# ------------------ (c) 融合FeedForward (可选) ------------------ 
class FusionFeedForward(nn.Module):
    """
    将自身特征和跨模态注意力得到的特征进行融合。
    这里做一个简单的 concat -> MLP -> 残差 + LN 示例
    """
    def __init__(self, embed_dim=768, hidden_dim=1024, dropout=0.1):
        super(FusionFeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.layernorm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_cross):
        """
        x: (B, seq, dim)
        x_cross: (B, seq, dim)
        """
        cat_x = torch.cat([x, x_cross], dim=-1)  # (B, seq, 2*dim)
        fused = self.ffn(cat_x)                 # (B, seq, dim)
        out = self.layernorm(x + self.dropout(fused))
        return out


# ------------------ (d) 将注意力 + 融合封装成一个模块 (可多层堆叠) ------------------ 
class CrossModalBlock(nn.Module):
    """
    一个Block包括: 双向注意力 -> 融合
    """
    def __init__(self, embed_dim=768, num_heads=4, hidden_dim=1024, dropout=0.1):
        super(CrossModalBlock, self).__init__()
        self.cross_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        self.fusion_eeg = FusionFeedForward(embed_dim, hidden_dim, dropout)
        self.fusion_img = FusionFeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x_eeg, x_img):
        # 双向注意力
        eeg_after_attn, img_after_attn = self.cross_attn(x_eeg, x_img)
        # 再融合 (这里演示: EEG 融合自己 + img_after_attn,   Img 融合自己 + eeg_after_attn)
        fused_eeg = self.fusion_eeg(x_eeg, img_after_attn)
        fused_img = self.fusion_img(x_img, eeg_after_attn)
        return fused_eeg, fused_img


# Image2EEG
class IE():
    def __init__(self, args, nsub):
        super(IE, self).__init__()
        self.args = args
        self.num_class = 200
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
        self.nSub = nsub

        self.start_epoch = 0
        self.eeg_data_path = '/mnt/bn/haopei-personal-v1/work/Data/Things-EEG2/Preprocessed_data_250Hz'
        self.img_data_path = '/mnt/bn/haopei-personal-v1/work/NICE-EEG-main/dnn_feature/'
        self.test_center_path = '/mnt/bn/haopei-personal-v1/work/NICE-EEG-main/dnn_feature/'
        self.pretrain = False


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.Enc_eeg = Enc_eeg().cuda()
        self.Proj_eeg = Proj_eeg().cuda()
        self.Proj_img = Proj_img().cuda()
        self.Enc_eeg = nn.DataParallel(self.Enc_eeg, device_ids=[i for i in range(len(gpus))]) # len(gpus
        self.Proj_eeg = nn.DataParallel(self.Proj_eeg, device_ids=[i for i in range(len(gpus))])
        self.Proj_img = nn.DataParallel(self.Proj_img, device_ids=[i for i in range(len(gpus))])

        # TODO
        # 3) 双向跨模态注意力 + 融合Block
        self.cross_modal_block = CrossModalBlock(embed_dim=768, 
                                                 num_heads=4, 
                                                 hidden_dim=1024, 
                                                 dropout=0.1).cuda()
        self.cross_modal_block = nn.DataParallel(self.cross_modal_block, device_ids=[i for i in range(len(gpus))])


        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('initial define done.')


    def get_eeg_data(self):
        args = parser.parse_args()

        train_data = []
        train_label = []
        test_data = []
        test_label = np.arange(200)

        train_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_training.npy', allow_pickle=True)
        train_data = train_data['preprocessed_eeg_data'] # (16540, 4, 63, 250)
        # 参数
        n_samples, n_repeats, n_channels, n_times = train_data.shape
        mask_ratio = args.ratio  # 20%通道
        n_mask = int(n_channels * mask_ratio)  # 要mask多少个通道

        # 随机 mask 每个样本
        for i in range(n_samples):
            mask_channels = np.random.choice(n_channels, n_mask, replace=False)
            train_data[i, :, mask_channels, :] = 0.0  # 或者用 np.nan 或其它方式掩盖

        train_data = np.mean(train_data, axis=1) # (16540, 63, 250)
        train_data = np.expand_dims(train_data, axis=1) # (16540, 1, 63, 250)

        test_data = np.load(self.eeg_data_path + '/sub-' + format(self.nSub, '02') + '/preprocessed_eeg_test.npy', allow_pickle=True) # test eeg
        test_data = test_data['preprocessed_eeg_data'] # (200, 80, 63, 250)
        test_data = np.mean(test_data, axis=1) # (200, 63, 250)
        test_data = np.expand_dims(test_data, axis=1) # (200, 1, 63, 250)

        return train_data, train_label, test_data, test_label

    def get_image_data(self):
        train_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_training.npy', allow_pickle=True)
        test_img_feature = np.load(self.img_data_path + self.args.dnn + '_feature_maps_test.npy', allow_pickle=True)

        train_img_feature = np.squeeze(train_img_feature) # (16540, 768)
        test_img_feature = np.squeeze(test_img_feature)  # (200, 768)

        return train_img_feature, test_img_feature
        
    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(self):
        
        self.Enc_eeg.apply(weights_init_normal)
        self.Proj_eeg.apply(weights_init_normal)
        self.Proj_img.apply(weights_init_normal)
        self.cross_modal_block.apply(weights_init_normal)


        # 可选：监控模型参数（这样 wandb 会自动记录梯度、权重直方图等）
        wandb.watch(self.Enc_eeg)
        wandb.watch(self.Proj_eeg)
        wandb.watch(self.Proj_img)
        wandb.watch(self.cross_modal_block)


        # train_eeg (16540, 1, 63, 250)   # test_eeg (200, 1, 63, 250)   # test_label  (200,)
        train_eeg, _, test_eeg, test_label = self.get_eeg_data()
        train_img_feature, _ = self.get_image_data()  # (16540, 768)
        test_center = np.load(self.test_center_path + 'center_' + self.args.dnn + '.npy', allow_pickle=True) # (200, 768)  === test image data

        train_image = train_img_feature


        train_eeg = torch.from_numpy(train_eeg)  # torch.Size([15800, 1, 63, 250])
        train_image = torch.from_numpy(train_image)  # torch.Size([15800, 768])

        dataset = torch.utils.data.TensorDataset(train_eeg, train_image)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_eeg = torch.from_numpy(test_eeg) # torch.Size([200, 1, 63, 250])

        test_center = torch.from_numpy(test_center) # torch.Size([200, 768])
        test_label = torch.from_numpy(test_label) # torch.Size([200])
        test_dataset = torch.utils.data.TensorDataset(test_eeg, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size_test, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), self.Proj_eeg.parameters(), self.Proj_img.parameters(),self.cross_modal_block.parameters()),lr=self.lr, betas=(self.b1, self.b2))

        num = 0
        best_loss_val = np.inf

        for e in range(self.n_epochs):
            in_epoch = time.time()

            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()
            self.cross_modal_block.train()


            total_correct = 0
            total_samples = 0

            for i, (eeg, img) in enumerate(self.dataloader):
                # eeg  torch.Size([1000, 1, 63, 250]) # img torch.Size([1000, 768])
                eeg = Variable(eeg.cuda().type(self.Tensor))
                # img = Variable(img.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor)) # torch.Size([1000, 768])

                labels = torch.arange(eeg.shape[0])  # used for the loss  # 1000
                labels = Variable(labels.cuda().type(self.LongTensor))

                # obtain the features
                eeg_features = self.Enc_eeg(eeg)  # torch.Size([1000, 1440])


                # project the features to a multimodal embedding space 
                eeg_features = self.Proj_eeg(eeg_features) # torch.Size([1000, 768])    Proj_eeg: 1440->768
                img_features = self.Proj_img(img_features)  # torch.Size([1000, 768])    Proj_img: 768->768


                # 3) 为了做Cross-Attention, 加一个序列长度维度: (B,1,768)
                eeg_features_3d = eeg_features.unsqueeze(1)
                img_features_3d = img_features.unsqueeze(1)

                # 4) 做双向注意力 + 融合
                #    输出仍是 (B,1,768), (B,1,768)
                fused_eeg, fused_img = self.cross_modal_block(eeg_features_3d, img_features_3d)

                # 融合后，如果继续对比学习，需要回到 (B,768)
                eeg_features = fused_eeg.squeeze(1)  # (B,768)
                img_features = fused_img.squeeze(1)  # (B,768)

                # cosine similarity as the logits
                logit_scale = self.logit_scale.exp()
                logits_per_eeg = logit_scale * eeg_features @ img_features.t() # torch.Size([1000, 1000])
                logits_per_img = logits_per_eeg.t()   # torch.Size([1000, 1000])

                
                loss_eeg = self.criterion_cls(logits_per_eeg, labels) # tensor(6.9683, device='cuda:0', grad_fn=<NllLossBackward0>) 给定 EEG 找到正确图像
                loss_img = self.criterion_cls(logits_per_img, labels) # tensor(6.9916, device='cuda:0', grad_fn=<NllLossBackward0>) 给定图像找到正确 EEG

                loss = (loss_eeg + loss_img) / 2 # tensor(6.9799, device='cuda:0', grad_fn=<DivBackward0>)

                # TODO  accuracy 
                preds_eeg = logits_per_eeg.argmax(dim=1)

                # print("preds_eeg", preds_eeg)
                # print("labels", labels)
                # 统计正确预测的数量和总样本数
                total_correct += (preds_eeg == labels).sum().item()
                total_samples += labels.size(0)



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 计算整个 dataloader 的准确率
            overall_acc = total_correct / total_samples
            print("Epoch {} train accuracy: {:.4f}".format(e + 1, overall_acc*100))
            # 记录到 wandb（可选）
            wandb.log({
                "epoch": e + 1,
                "train_loss": loss.item(),
                "train_acc": overall_acc
            })


            torch.save(self.Enc_eeg.state_dict(), './model_2_/' + f"Enc_eeg_sub{self.nSub:02}.pth")
            torch.save(self.Proj_eeg.state_dict(), './model_2_/' + f"Proj_eeg_sub{self.nSub:02}.pth")
            torch.save(self.Proj_img.state_dict(), './model_2_/' + f"Proj_img_sub{self.nSub:02}.pth")
            torch.save(self.cross_modal_block.state_dict(), './model_2_/' + f"cross_modal_block_sub{self.nSub:02}.pth")



        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        self.Enc_eeg.load_state_dict(torch.load('./model_2_/' + f"Enc_eeg_sub{self.nSub:02}.pth"), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model_2_/' + f"Proj_eeg_sub{self.nSub:02}.pth"), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model_2_/' + f"Proj_img_sub{self.nSub:02}.pth"), strict=False)
        self.cross_modal_block.load_state_dict(torch.load('./model_2_/' + f"cross_modal_block_sub{self.nSub:02}.pth"), strict=False)

        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()
        self.cross_modal_block.eval()

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.type(self.Tensor))  # torch.Size([200, 1, 63, 250])
                tlabel = Variable(tlabel.type(self.LongTensor)) # torch.Size([200])
                all_center = Variable(all_center.type(self.Tensor))       # torch.Size([200, 768])      

                tfea = self.Proj_eeg(self.Enc_eeg(teeg)) # torch.Size([200, 768])
                tfea = tfea / tfea.norm(dim=1, keepdim=True) # torch.Size([200, 768])

                # TODO
                # all_center = self.Proj_img(all_center) # torch.Size([, 768])

                # 3) 进行跨模态注意力
                tfea = tfea.unsqueeze(1)  # (200, 1, 768)
                all_center = all_center.unsqueeze(1)    # (200, 1, 768)  (可选，看是否对 all_center 也进行跨模态处理)

                fused_tfea_eeg, fused_tcenter = self.cross_modal_block(tfea, all_center)

                # 4) 恢复回 (200, 768) 形状
                tfea = fused_tfea_eeg.squeeze(1)  # (200, 768)
                all_center = fused_tcenter.squeeze(1)    # (200, 768) 

                similarity = tfea @ all_center.t().softmax(dim=-1)  # no use 100? # torch.Size([200, 200])
                num, indices = similarity.topk(5) # indices: torch.Size([200, 5])   [169, 155,  77, 170,  59],

                tt_label = tlabel.view(-1, 1) # torch.Size([200, 1])
                total += tlabel.size(0) # 200

                top1 += (tt_label == indices[:, :1]).sum().item()
                top3 += (tt_label == indices[:, :3]).sum().item()
                top5 += (tt_label == indices).sum().item()

                # print("Top1 correct count:", indices[:, :1])
                # print("Top3 correct count:", indices[:, :3])
                # print("Top5 correct count:", indices)
                # print("Top5 similarity:", num)


            
            top1_acc = float(top1) / float(total)
            top3_acc = float(top3) / float(total)
            top5_acc = float(top5) / float(total)
        
        print('The test Top1-%.6f, Top3-%.6f, Top5-%.6f' % (top1_acc, top3_acc, top5_acc))

        
        return top1_acc, top3_acc, top5_acc
        # writer.close()


def main():
    args = parser.parse_args()

    # 初始化 wandb，设定项目名称和配置参数
    wandb.login(key="91d91f7d8ff49f7ab098b924d88696d1658c922f")

    num_sub = args.num_sub   
    cal_num = 0
    aver = []
    aver3 = []
    aver5 = []

    # 获取当前日期和时间
    current_time = datetime.datetime.now()
    # 格式化时间为 "月日时分" 格式，例如 "03200945" 表示 3月20日09:45
    time_str = current_time.strftime("%m%d%H%M%S")

    # seed_n = np.random.randint(args.seed)
    seed_n = args.seed
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'all seed is {seed_n}')
    print(f'ratio is {args.ratio}')
    for i in range(num_sub):  # 被试者编号从 1 开始
        run = wandb.init(
            project=f"NICE_wo_val_{time_str}_{seed_n}",  # 使用动态生成的时间戳
            config=vars(args),
            name=f"Subject_{i+1}",
            reinit=True
        )

        cal_num += 1
        starttime = datetime.datetime.now()



        print('Subject %d' % (i+1))
        ie = IE(args, i + 1) # 针对当前被试 i+1 初始化模型

        Acc, Acc3, Acc5 = ie.train() # 训练该被试的模型
        print('THE BEST ACCURACY IS ' + str(Acc))


        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        aver.append(Acc)
        aver3.append(Acc3)
        aver5.append(Acc5)

    aver.append(np.mean(aver))
    aver3.append(np.mean(aver3))
    aver5.append(np.mean(aver5))

    column = np.arange(1, cal_num+1).tolist()
    column.append('ave')
    pd_all = pd.DataFrame(columns=column, data=[aver, aver3, aver5])
    print("pd_all",pd_all)
    print("seed_n",seed_n)
    pd_all.to_csv(result_path + str(seed_n) + '-'+ str(time_str) + 'woval_2_.csv')



    
if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))