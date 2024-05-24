import pandas as pd
from transformers import AutoTokenizer, AutoModel


df = pd.read_csv('data/hotseatch_history.csv')

sample_news = df[df['日期']=='2020-11-24'][['热搜词条', '标签（时政、科技、科普、娱乐、体育、社会讨论/话题、时事、经济）']]
print(len(sample_news))

sample_inputs = {}
for li, lab in sample_news.values.tolist():
    if lab not in sample_inputs:
        sample_inputs[lab] = [li]
    else:
        sample_inputs[lab].append(li)
sample_inputs = list(list(li) for li in sample_inputs.items())
for sample in sample_inputs:
    sample[1] = ','.join(sample[1])

sample_inputs = '|'.join([':'.join(li) for li in sample_inputs])

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/kaggle/input/chatglm3-6b", trust_remote_code=True).half().cuda()
model = model.eval()

response, history = model.chat(tokenizer, "假设你是一名专业的新闻助手，你将根据我给你提供的每日新闻回答相应的问题", history=[])
print(response)

response, history = model.chat(tokenizer, 
                               "给你一天的新闻汇总文档，其中，每一类新闻以'类别：新闻清单'的形式呈现，新闻类与类之间使用'|'分隔。以下是今天的新闻汇总，请你告诉我今天发生了什么科技类新闻："
                               +sample_inputs, 
                               history=history)
print(response)

