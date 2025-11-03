"""
留一交叉验证：
- 总共有10个受试者，
- 每轮将其中1个受试者作为测试集，其余9个受试者的训练数据联合起来作为训练集，
- 训练结束后，在测试受试者的测试数据上评估性能，
- 循环所有受试者，并求平均结果。
"""

import wandb
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
# debugpy.listen(('0.0.0.0', 5200))  # 监听所有地址的5200端口
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # 等待VSCode连接

# 指定GPU
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
result_path = '/mnt/bn/haopei-personal-v1/work/NICE-EEG-main/' 

parser = argparse.ArgumentParser(description='留一交叉验证：基于EEG和图像特征的多模态对齐模型')
parser.add_argument('--dnn', default='clip', type=str)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--num_sub', default=10, type=int,
                    help='总共受试者数量（默认10）')
parser.add_argument('-batch_size', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch大小')
parser.add_argument('--seed', default=2023, type=int, help='随机种子')


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

# EEG编码器：先通过PatchEmbedding，再平坦化
class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

# EEG投影头：将1440维映射到768维
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

# 图像投影头（本例中对图像特征维度保持不变）
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
    



# 定义模型类：支持多受试者训练（训练集来自多个受试者）和单一测试受试者
class IE():
    def __init__(self, args, train_subs, test_sub):
        """
        args: 参数
        train_subs: list，训练受试者ID列表（如[1,2,3,...,10]中去掉测试者）
        test_sub: int，测试受试者ID
        """
        self.args = args
        self.train_subs = train_subs
        self.test_sub = test_sub
        
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


         # TODO
        # 3) 双向跨模态注意力 + 融合Block
        self.cross_modal_block = CrossModalBlock(embed_dim=768, 
                                                 num_heads=4, 
                                                 hidden_dim=1024, 
                                                 dropout=0.1).cuda()
        self.cross_modal_block = nn.DataParallel(self.cross_modal_block, device_ids=[i for i in range(len(gpus))])

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.centers = {}
        print('模型初始化完成。')

    # 加载训练受试者的EEG数据，并将多个受试者的数据合并
    def get_eeg_train_data(self):
        all_train = []
        for sub in self.train_subs:
            file = os.path.join(self.eeg_data_path, 'sub-' + format(sub, '02'), 'preprocessed_eeg_training.npy')
            data = np.load(file, allow_pickle=True)
            data = data['preprocessed_eeg_data']  # 原始形状 (16540, 4, 63, 250)
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
        self.cross_modal_block.apply(weights_init_normal)


        # 用wandb监控模型参数
        wandb.watch(self.Enc_eeg)
        wandb.watch(self.Proj_eeg)
        wandb.watch(self.Proj_img)
        wandb.watch(self.cross_modal_block)

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

        # 定义优化器
        self.optimizer = torch.optim.Adam(itertools.chain(self.Enc_eeg.parameters(), 
                                                          self.Proj_eeg.parameters(),
                                                         self.Proj_img.parameters(),
                                                        self.cross_modal_block.parameters()),lr=self.lr, betas=(self.b1, self.b2))


        best_loss_val = np.inf

        for e in range(self.n_epochs):
            self.Enc_eeg.train()
            self.Proj_eeg.train()
            self.Proj_img.train()
            self.cross_modal_block.train()


            total_correct = 0
            total_samples = 0

            for i, (eeg, img) in enumerate(self.dataloader):
                eeg = Variable(eeg.cuda().type(self.Tensor))
                img_features = Variable(img.cuda().type(self.Tensor))
                labels = torch.arange(eeg.shape[0]).cuda().type(self.LongTensor)

                # 前向传播：EEG编码及投影
                eeg_features = self.Enc_eeg(eeg)   # (B, 1440)
                eeg_features = self.Proj_eeg(eeg_features)  # (B, 768)
                img_features = self.Proj_img(img_features)  # (B, 768)


                # 3) 为了做Cross-Attention, 加一个序列长度维度: (B,1,768)
                eeg_features_3d = eeg_features.unsqueeze(1)
                img_features_3d = img_features.unsqueeze(1)

                # 4) 做双向注意力 + 融合
                #    输出仍是 (B,1,768), (B,1,768)
                fused_eeg, fused_img = self.cross_modal_block(eeg_features_3d, img_features_3d)

                # 融合后，如果继续对比学习，需要回到 (B,768)
                eeg_features = fused_eeg.squeeze(1)  # (B,768)
                img_features = fused_img.squeeze(1)  # (B,768)


                # 计算余弦相似度，并乘以logit_scale
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
            print("Epoch {} train accuracy: {:.4f}%".format(e + 1, overall_acc*100))

            # 记录到 wandb（可选）
            wandb.log({
                "epoch": e + 1,
                "train_loss": loss.item(),
                "train_acc": overall_acc
            })

            torch.save(self.Enc_eeg.state_dict(), './model_cross/Enc_eeg_sub' + format(self.test_sub, '02') + '.pth')
            torch.save(self.Proj_eeg.state_dict(), './model_cross/Proj_eeg_sub' + format(self.test_sub, '02') + '.pth')
            torch.save(self.Proj_img.state_dict(), './model_cross/Proj_img_sub' + format(self.test_sub, '02') + '.pth')
            torch.save(self.cross_modal_block.state_dict(), './model_cross/cross_modal_block' + format(self.test_sub, '02') + '.pth')




        # * test part
        all_center = test_center
        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        # 测试阶段：加载最佳模型并在测试集上计算Top-1, Top-3, Top-5准确率

        self.Enc_eeg.load_state_dict(torch.load('./model_cross/Enc_eeg_sub' + format(self.test_sub, '02') + '.pth'), strict=False)
        self.Proj_eeg.load_state_dict(torch.load('./model_cross/Proj_eeg_sub' + format(self.test_sub, '02') + '.pth'), strict=False)
        self.Proj_img.load_state_dict(torch.load('./model_cross/Proj_img_sub' + format(self.test_sub, '02') + '.pth'), strict=False)
        self.cross_modal_block.load_state_dict(torch.load('./model_cross/cross_modal_block' + format(self.test_sub, '02') + '.pth'), strict=False)


        self.Enc_eeg.eval()
        self.Proj_eeg.eval()
        self.Proj_img.eval()
        self.cross_modal_block.eval()


        total = 0
        top1 = 0
        top3 = 0
        top5 = 0

        with torch.no_grad():
            for i, (teeg, tlabel) in enumerate(self.test_dataloader):
                teeg = Variable(teeg.cuda().type(self.Tensor))
                tlabel = Variable(tlabel.cuda().type(self.LongTensor))
                all_center = Variable(test_center.cuda().type(self.Tensor))  # (200, 768)

                tfea = self.Proj_eeg(self.Enc_eeg(teeg))  # (200, 768)
                tfea = tfea / tfea.norm(dim=1, keepdim=True)


                # 3) 进行跨模态注意力
                tfea = tfea.unsqueeze(1)  # (200, 1, 768)
                all_center = all_center.unsqueeze(1)    # (200, 1, 768)  (可选，看是否对 all_center 也进行跨模态处理)

                fused_tfea_eeg, fused_tcenter = self.cross_modal_block(tfea, all_center)

                # 4) 恢复回 (200, 768) 形状
                tfea = fused_tfea_eeg.squeeze(1)  # (200, 768)
                all_center = fused_tcenter.squeeze(1)    # (200, 768) 

                similarity = tfea @ all_center.t().softmax(dim=-1)  # no use 100? # torch.Size([200, 200])
                num, indices = similarity.topk(5) # indices: torch.Size([200, 5])   [169, 155,  77, 170,  59],

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

    # 对每个受试者作为测试集，训练集为其余9个


  # 获取当前日期和时间
    current_time = datetime.datetime.now()
    # 格式化时间为 "月日时分" 格式，例如 "03200945" 表示 3月20日09:45
    time_str = current_time.strftime("%m%d%H%M%S")
    wandb.login(key="91d91f7d8ff49f7ab098b924d88696d1658c922f")

    seed_n = args.seed
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'all seed is {seed_n}')


    for test_sub in range(1, num_sub+1):
        train_subs = [s for s in range(1, num_sub+1) if s != test_sub]
        run = wandb.init(project=f"cross_subject_974_woval_{time_str}", config=vars(args), name=f"TestSub_{test_sub}", reinit=True)
        print('Leave-one-out: Test subject {}'.format(test_sub))
        starttime = datetime.datetime.now()

        # # 设置随机种子
        # seed_n = args.seed
        # random.seed(seed_n)
        # np.random.seed(seed_n)
        # torch.manual_seed(seed_n)
        # torch.cuda.manual_seed(seed_n)
        # torch.cuda.manual_seed_all(seed_n)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # print(f'all seed is {seed_n}')

        ie = IE(args, train_subs, test_sub)
        Acc, Acc3, Acc5 = ie.train()
        print('Subject {} Test Top1 Acc: {:.4f}'.format(test_sub, Acc))
        endtime = datetime.datetime.now()
        print('Subject {} duration: {}'.format(test_sub, endtime - starttime))
        all_top1.append(Acc)
        all_top3.append(Acc3)
        all_top5.append(Acc5)
        wandb.finish()

    # 计算平均结果
    avg_top1 = np.mean(all_top1)
    avg_top3 = np.mean(all_top3)
    avg_top5 = np.mean(all_top5)
    print('Overall Test Accuracy:')
    print("all_top1",all_top1)
    print("all_top5",all_top5)
    print('Average Top1: {:.6f}, Top3: {:.6f}, Top5: {:.6f}'.format(avg_top1, avg_top3, avg_top5))

    # 保存结果到CSV文件
    results = {
        'Subject': list(range(1, num_sub+1)) + ['Average'],
        'Top1': all_top1 + [avg_top1],
        'Top3': all_top3 + [avg_top3],
        'Top5': all_top5 + [avg_top5],
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(result_path, f'result{seed_n}_{time_str}_cross.csv'), index=False)
    print("Results saved.")

if __name__ == "__main__":
    main()
