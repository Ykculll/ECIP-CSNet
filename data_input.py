import numpy as np
import torch
import os
import scipy.io
from torch.utils.data import Dataset, DataLoader

# 数据路径
SEEDIV_data_dir = "../../../datasets/SEED-IV/eeg_feature_smooth"

# 情绪标签
seesion1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
seesion2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
seesion3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = [seesion1_label, seesion2_label, seesion3_label]

# 每类 trial 中，最后两个为测试 trial
trials_seesion1 = [[3,5,6,8,20,22],[0,7,9,11,12,13],[1,4,10,14,16,17],[2,15,18,19,21,23]]
trials_seesion2 = [[3,4,6,13,18,20],[1,14,15,17,21,23],[0,5,7,10,12,16],[2,8,9,11,19,22]]
trials_seesion3 = [[11,15,18,19,21,23],[0,3,7,8,10,22],[1,2,9,12,16,20],[4,5,6,13,14,17]]
trials = [trials_seesion1, trials_seesion2, trials_seesion3]


class SEEDIV_dataset(Dataset):
    def __init__(self, SEEDIV_data_dir, subj_num, session_num, is_train=None):
        super(SEEDIV_dataset, self).__init__()
        self.is_train = is_train
        self.session_num = session_num
        self.sample_list = []
        self.sample_label_list = []

        # 找到 subject 文件路径
        session_dir = os.path.join(SEEDIV_data_dir, str(self.session_num))
        for subj_file in os.listdir(session_dir):
            subj = int(subj_file.split("_")[0])
            if subj == subj_num:
                subj_file_name = subj_file
                break
        file_path = os.path.join(session_dir, subj_file_name)

        # 构建测试 trial 列表（每类最后两个）
        self.test_Trails = []
        for i in range(4):
            self.test_Trails.append(trials[self.session_num - 1][i][4])
            self.test_Trails.append(trials[self.session_num - 1][i][5])

        # 遍历每个 trial
        for trial in range(24):
            X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial + 1)]
            y = int(labels[self.session_num - 1][trial])

            for t in range(X_de.shape[1]):
                x_de = torch.tensor(X_de[:, t, :]).float()  # [62, 5]
                x_de = (x_de - x_de.mean()) / (x_de.std() + 1e-6)   # 标准化

                if self.is_train is None:
                    self.sample_list.append(x_de)
                    self.sample_label_list.append(y)
                elif self.is_train is True and trial not in self.test_Trails:
                    self.sample_list.append(x_de)
                    self.sample_label_list.append(y)
                elif self.is_train is False and trial in self.test_Trails:
                    self.sample_list.append(x_de)
                    self.sample_label_list.append(y)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_list[index], self.sample_label_list[index]


def getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=8, is_train=None, is_drop_last=False, is_shuffle=True, num_workers=0):
    gen_dataset = SEEDIV_dataset(SEEDIV_data_dir, subj_num, session_num, is_train)
    return DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        num_workers=num_workers,
        drop_last=is_drop_last,
    )


if __name__ == "__main__":
    # 测试getloader能否正常加载数据
    batch_size = 32
    data_loader = getloader(SEEDIV_data_dir, subj_num=1, session_num=1, batch_size=batch_size, is_train=True)
    for step, (data, labels) in enumerate(data_loader):
        print("Step:", step)
        print("x shape:", data.shape)    # [B, 62, 5]
        print("y shape:", labels.shape)  # [B]
        break
