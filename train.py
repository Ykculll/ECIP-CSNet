import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random

from model import Net
from data_input import getloader


# 设置随机数种子
def seed_torch(seed=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

# psnr计算用于预训练
def calculate_psnr(original, reconstructed):
    
    assert original.shape == reconstructed.shape, "输入维度必须一致"

    # 展平除 batch 之外的维度
    B = original.size(0)
    original = original.view(B, -1)
    reconstructed = reconstructed.view(B, -1)

    mse = F.mse_loss(reconstructed, original, reduction='none')  # [B, N]
    mse = mse.mean(dim=1)  # 每个样本的 MSE，形状 [B]

    #动态计算data_range
    data_min = original.min()
    data_max = original.max()
    data_range = data_max - data_min

    # 避免 log(0)
    mse = torch.clamp(mse, min=1e-10)

    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))  # [B]
    return psnr.mean().item()  # 返回 batch 平均


def cre_prolog(session_num=None):
    current_script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script_path)
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log')

    # 创建主日志目录
    if not os.path.exists(pro_log_dir_path):
        os.makedirs(pro_log_dir_path)
        print(f"Directory '{pro_log_dir_path}' created.")

    # 创建 session 子目录
    if session_num is not None:
        session_dir = os.path.join(pro_log_dir_path, f'session{session_num}')
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            print(f"Directory '{session_dir}' created.")
        return session_dir

    return pro_log_dir_path


def cre_prolog_pre(session_num=None):
    current_script_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_script_path)
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log_pre')

    # 创建主日志目录
    if not os.path.exists(pro_log_dir_path):
        os.makedirs(pro_log_dir_path)
        print(f"Directory '{pro_log_dir_path}' created.")

    # 创建 session 子目录
    if session_num is not None:
        session_dir = os.path.join(pro_log_dir_path, f'session{session_num}')
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            print(f"Directory '{session_dir}' created.")
        return session_dir

    return pro_log_dir_path

# 配置参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
SEEDIV_data_dir = '../datasets/SEED-IV/eeg_feature_smooth'


# 通道一致性损失
def channel_kl_consistency_loss(kl_std, labels, eps=1e-8):

    device = kl_std.device
    unique_labels = torch.unique(labels)
    total_loss = 0.0
    valid_classes = 0

    for c in unique_labels:
        mask = (labels == c)
        p_cls = kl_std[mask]          # [Nc, 62]
        Nc = p_cls.size(0)

        if Nc <= 1:
            continue

        # batch 内 reference（不反传梯度）
        p_ref = p_cls.mean(dim=0, keepdim=True).detach()  # [1, 62]

        # KL(p_i || p_ref)
        kl = p_cls * (torch.log(p_cls + eps) - torch.log(p_ref + eps))
        kl = kl.sum(dim=1).mean()  # 对样本、通道求均值

        total_loss += kl
        valid_classes += 1

    if valid_classes == 0:
        return kl_std.sum() * 0.0

    return total_loss / valid_classes


# 训练流程
def train_subject(subj_num=1, session_num=1, batch_size=64, lr=1e-3, epochs=20):

    exp_dir = cre_prolog(session_num)
    exp_dir_pre = cre_prolog_pre(session_num)
    log_file = os.path.join(exp_dir, f'log_session{session_num}.txt')


    net = Net(fea_dim=5, channel_dim=62, num_classes=4, topk_ratio=0.5, hidden_dim=32).to(device)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    best_val_acc = 0.0
    best_epoch = 0
    lowest_loss = 9999

    best_model_path = f"{exp_dir}/best_classifier_subj{subj_num}_session{session_num}.pth"
    best_premodel_path = f"{exp_dir_pre}/best_loss_pre_subj{subj_num}_session{session_num}.pth"

    # 训练集、验证集
    tra_loader = getloader(SEEDIV_data_dir, subj_num=subj_num, session_num=session_num, batch_size=batch_size, is_train=True, is_drop_last=True)
    val_loader = getloader(SEEDIV_data_dir, subj_num=subj_num, session_num=session_num, batch_size=batch_size, is_train=False, is_drop_last=False)
    # 预训练

    for epoch in range(epochs):
        net.train()
        
        tra_loss = 0
        psnr_all = 0
        num_batch = 0

        for batch, label in tra_loader:
            x = batch.to(device)
            x_recon, kl_std= net(x, mode='pretrain')

            # 重构损失  
            loss_mse = mse_loss(x_recon, x)

            loss = loss_mse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tra_loss += loss.item() * x.size(0)
            psnr_all += calculate_psnr(x, x_recon)
            num_batch += 1

        tra_loss /= len(tra_loader.dataset)
        psnr = psnr_all / num_batch

        # 保存预训练模型
        if lowest_loss > tra_loss:
            lowest_loss = tra_loss
            torch.save(net.state_dict(), best_premodel_path)

        log_line = f'[Pretrain] Epoch {epoch + 1}/{epochs}, Loss: {tra_loss:.4f}, best_loss: {lowest_loss:.4f}, psnr: {psnr:.4f}\n'
        with open(log_file, 'a') as f:
            f.write(log_line)
        print(log_line.strip())
       
        # 提前停止机制
        if tra_loss < 0.0001 or psnr > 45:
            print(f'预训练提前停止，loss达到 {tra_loss:.4f},psnr达到 {psnr:.4f}')
            break

    # 加载预训练模型
    net.load_state_dict(torch.load(best_premodel_path))

    # 主训练

    for epoch in range(epochs):
        net.train()
        
        tra_loss = 0
        tra_acc = 0
        total = 0
        for batch, label in tra_loader:
            x = batch.to(device)
            label = label.to(device)

            # 向前传播
            x_recon, pred, kl_std= net(x, mode = 'maintrain')

            # 重构损失  
            loss_mse = mse_loss(x_recon, x)

            # 分类损失
            loss_cls = ce_loss(pred, label)

            # 通道一致性损失
            loss_kl = channel_kl_consistency_loss(kl_std, label)

            # 总loss
            loss = loss_cls + 5 * loss_kl + 0.05 * loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tra_loss += loss.item() * x.size(0)
            _, predicted = torch.max(pred, 1)
            tra_acc += (predicted == label).sum().item()
            total += label.size(0)
        tra_loss = tra_loss / len(tra_loader.dataset)
        acc = tra_acc / total if total > 0 else 0
        
        # 验证集评估
        net.eval()
        
        val_acc = 0
        val_total = 0
        with torch.no_grad():
            for batch, label in val_loader:
                x = batch.to(device)
                label = label.to(device)
                x_recon, pred, _= net(x, mode='maintrain')
                _, predicted = torch.max(pred, 1)
                val_acc += (predicted == label).sum().item()
                val_total += label.size(0)
        val_acc = val_acc / val_total if val_total > 0 else 0
        
        
        log_line = f'[Train] Epoch {epoch + 1}/{epochs}, Loss: {tra_loss:.4f}, tra_Acc: {acc:.4f}, val_Acc: {val_acc:.4f}, Best_acc: {best_val_acc:.4f}, Best_Epoch: {best_epoch}\n'
        with open(log_file, 'a') as f:
            f.write(log_line)
        print(log_line.strip())

        # 保存val_acc最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(net.state_dict(), best_model_path)

        # 提前停止机制
        if val_acc == 1:
            print(f'训练提前停止，验证准确率达到100%')
            break
            
    return best_val_acc, best_epoch


if __name__ == '__main__':
    batch_size = 16
    lr = 0.0025
    epochs = 200
    session_num = 3  # 选择使用第几个session的数据
    
    exp_dir = cre_prolog(session_num)
    log_file = os.path.join(exp_dir, f'log_session{session_num}.txt')

    # 开始训练
    best_Acc = []
    with open(log_file, 'a') as f:
        f.write('\n==================== Training Start ====================\n')
        f.write(f'batch_size: {batch_size}, epochs: {epochs}, lr: {lr}, session_num: {session_num}\n')
    for subj in range(1, 16):  # 可根据实际subject数调整
        print(f'------------------start subject:{subj} session:{session_num}---------------------')
        with open(log_file, 'a') as f:
            f.write(f'\n------------------start subject:{subj} session:{session_num}---------------------\n')
        best_model_acc, best_epoch = train_subject(subj_num=subj, session_num=session_num, batch_size=batch_size, lr=lr,
                                                   epochs=epochs)
        best_Acc.append(best_model_acc)

    k = 0
    result_string = ''
    for i in best_Acc:
        k = k + 1
        str1 = f'subject{k} acc:{i} \n'
        result_string = ''.join([result_string, str1])

    # 统计结果
    mean_acc = np.mean(best_Acc)
    sta = np.std(best_Acc, ddof=1)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str])

    with open(log_file, 'a') as f:
        f.write(result_string)
    print(result_string)
