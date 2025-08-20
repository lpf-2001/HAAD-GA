import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
     


class MyLabelEncoder(LabelEncoder):
    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = np.unique(y) # pd.Series(y).unique()
        self.classes_.sort()  # sort for same result
        return self

def LabelEncode(label_str):
    y = label_str # 获得labelY，shape=[B, 1]
    y = [str(i) for i in y]  # 统一变成str
    enc = MyLabelEncoder()
    enc.fit(y)
    y__encode = enc.transform(y)  # 实现数据集label映射
    return y__encode
# def sample_traces(x, y, N):
#     num_classes = len(np.unique(y))
#     train_index = []

#     for c in range(num_classes):
#         idx = np.where(y == c)[0]
#         idx = np.random.choice(idx, min(N, len(idx)), False)
#         train_index.extend(idx)

#     train_index = np.array(train_index)
#     np.random.shuffle(train_index)

#     x_train = x[train_index]
#     y_train = y[train_index]
#     remaining_indices = np.array([i for i in range(len(x)) if i not in train_index])
#     return x_train, y_train, remaining_indices
def sample_traces(x, y,train_N,test_N):
    N=train_N+test_N
    num_classes = len(np.unique(y))
    test_index = []
    train_index = []
    climp_index = []
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        idx = np.random.choice(idx, min(N, len(idx)), False)
        climp_index.extend(idx)
        train_index.extend(idx[:train_N])
        test_index.extend(idx[train_N:])
        
    
    climp_index = np.array(climp_index)
    train_index = np.array(train_index)
    test_index = np.array(test_index)

    np.random.shuffle(train_index)
    np.random.shuffle(test_index)
    
    print('train_index.shape:',train_index.shape)
    print('test_index.shape: ',test_index.shape)

    remaining_indices = np.array([i for i in range(len(x)) if i not in climp_index])
    return train_index, test_index, remaining_indices

df_DataPath='/home/xuke/zpc/Security/DataSet/DF/DF_close-world.npz'
df_SavaPath='/home/xuke/zpc/Security/DataSet/DF/'

awf2_DataPath='/home/xuke/zpc/Security/DataSet/AWF/awf2.npz'
awf2_SavaPath='/home/xuke/zpc/Security/DataSet/AWF/'

split_name='df'

if  split_name=="df":
    ori_data_path=df_DataPath
    SavePath=df_SavaPath
elif split_name=="awf":
    ori_data_path=awf2_DataPath
    SavePath=awf2_SavaPath
print(f'split {split_name} dataset')

np.random.seed(42)
data = np.load(ori_data_path, allow_pickle=True)
x = data["data"]  # 257500
y = data["labels"]  # 103
print(len(np.unique(y)))
y=LabelEncode(y)
print(len(np.unique(y)))

train_Num=200
test_Num=100

index_train, index_test, index_remaining = sample_traces(x, y, train_Num,test_Num)

x_train, y_train = x[index_train], y[index_train]
x_test, y_test = x[index_test], y[index_test]
x_re, y_re = x[index_remaining], y[index_remaining]


np.savez_compressed(SavePath+f'{split_name}_small200.npz', data=x_train, labels=y_train)
np.savez_compressed(SavePath+f'{split_name}_small100_test.npz', data=x_test, labels=y_test)
np.savez_compressed(SavePath+f'{split_name}_remain_small200and100.npz', data=x_re, labels=y_re)


print('done')


