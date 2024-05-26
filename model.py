import torch
import torch.nn as nn
from transformers import BertModel, AutoModel

class BertLstmModel(nn.Module):
    def __init__(self, bert_config, num_labels, hidden_size=768, lstm_hidden_size=256, num_layers=2):
        super(BertLstmModel, self).__init__()

        # 加载预训练的 BERT 模型
        # self.bert = BertModel.from_pretrained(bert_config)
        self.bert = AutoModel.from_pretrained(bert_config)

        # LSTM 层
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(lstm_hidden_size * 2, num_labels)
        
        self.Dropout = nn.Dropout(p=0.3)
        
        # 冻结 BERT 模型的前几层
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # for param in self.bert.encoder.layer[-1].parameters():
        #     param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        # BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 取最后一层的输出
        bert_output = outputs.last_hidden_state
        
        bert_output = self.Dropout(bert_output)

        # LSTM 层
        lstm_output, _ = self.lstm(bert_output)

        # 取最后一层的输出
        lstm_output = lstm_output[:, -1, :]
        
        lstm_output = self.Dropout(lstm_output)

        # 全连接层
        logits = self.fc(lstm_output)

        return logits
