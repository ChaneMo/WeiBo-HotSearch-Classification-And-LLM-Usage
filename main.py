import numpy as np # linear algebra
import pandas as pd
from model import BertLstmModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from transformers import BertTokenizer, AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import torchmetrics
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.optim import AdamW
import collections


df = pd.read_csv('data/weibo-hot-search-labeled.csv')
df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'] = df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'].apply(lambda x: x.strip())
print(df.head())

cnt = collections.Counter(df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'])
print('标签分布：', cnt)

texts = df['热搜词条'].tolist()
labels = [li.strip() for li in df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'].tolist()]
# 将标签转换为数值类型
encoder = LabelEncoder()
encoder.fit(labels)
labels = encoder.transform(labels)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, stratify=labels, random_state=2024)

train_df = pd.DataFrame(X_train, columns=['热搜词条'])
train_df['标签'] = y_train

# 定义随机过采样对象
ros = RandomOverSampler()
# 对数据进行随机过采样
X_resampled, y_resampled = ros.fit_resample(train_df.drop('标签', axis=1), train_df['标签'])

# 将欠采样后的样本转换为DataFrame格式
resampled_df = pd.DataFrame(X_resampled, columns=['热搜词条'])
resampled_df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'] = y_resampled
print(resampled_df.head())

# 打印欠采样后的样本数量
print('Original dataset shape:', len(X_train))
print('Resampled dataset shape:', resampled_df.shape)

X_train, y_train = resampled_df['热搜词条'].tolist(), resampled_df['标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）'].tolist()


# 加载预训练的 BERT 模型和配置文件
bert_config = 'bert-base-chinese'
# bert_config = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = AutoTokenizer.from_pretrained(bert_config)
model = BertLstmModel(bert_config, num_labels=8)

# 定义超参数
batch_size = 32
max_length = 15
num_epochs = 5

# 冻结 BERT 模型的前几层
for param in model.parameters():
    param.requires_grad = False
# 仅运行bert最后一层参与权重更新
for param in model.encoder.layer[-1].parameters():
    param.requires_grad = True

# 创建 BertLstmModel 模型
device = torch.device('cuda')
model.to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=max_length)
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                               torch.tensor(train_encodings['attention_mask']),
                                               torch.tensor(y_train))

valid_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=max_length)
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_encodings['input_ids']),
                                               torch.tensor(valid_encodings['attention_mask']),
                                               torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)


learning_rate = 1e-3
# 定义优化器和学习率调度器
optimizer = AdamW([
    {'params': model.lstm.parameters(), 'lr': 5e-5},
    {'params': model.fc.parameters(), 'lr': 5e-5},
    {'params': model.bert.encoder.layer[-1].parameters(), 'lr': 5e-5}
], lr=learning_rate)
# 定义损失函数
criterion = nn.CrossEntropyLoss()
metric = torchmetrics.Precision(task='multiclass', num_classes=8)
metric.to(device)

# 训练模型
for epoch in range(num_epochs):
    # 训练模式
    model.train()

    # 训练损失和评价指标
    train_loss = 0.0
    train_metric = 0.0

    for batch in train_loader:
        # 将数据传入 GPU
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[-1].to(device)

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # 计算损失和评价指标
        loss = criterion(logits, label)
        metric_value = metric(torch.argmax(logits, dim=1), label)

        # 反向传播和更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累计损失和评价指标
        train_loss += loss.item() * input_ids.size(0)
        train_metric += metric_value * input_ids.size(0)

    # 计算平均损失和评价指标
    train_loss /= len(train_dataset)
    train_metric /= len(train_dataset)

    # 验证模式
    model.eval()

    # 验证损失和评价指标
    valid_loss = 0.0
    valid_metric = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            # 将数据传入 GPU
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            label = batch[-1].to(device)

            # 前向传播
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # 计算损失和评价指标
            loss = criterion(logits, label)
            metric_value = metric(torch.argmax(logits, dim=1), label)

            # 累计损失和评价指标
            valid_loss += loss.item() * input_ids.size(0)
            valid_metric += metric_value * input_ids.size(0)

        # 计算平均损失和评价指标
        valid_loss /= len(valid_dataset)
        valid_metric /= len(valid_dataset)

    # 打印训练日志
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train prec: {:.4f}, Valid Loss: {:.4f}, Valid prec: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, train_metric, valid_loss, valid_metric))

# 测试模式
model.eval()

# 预测结果和真实标签
y_pred = []
y_true = []

with torch.no_grad():
    for batch in valid_loader:
        # 将数据传入 GPU
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)

        # 前向传播
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # 计算预测结果和真实标签
        pred = torch.argmax(logits, dim=1)
        y_pred.extend(pred.cpu().numpy())
        y_true.extend(label.cpu().numpy())


# 计算准确率和召回率
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

# 输出准确率和召回率
print('Precision: {:.4f}'.format(precision))
print('Recall: {:.4f}'.format(recall))
print('F1: {:.4f}'.format(f1))

