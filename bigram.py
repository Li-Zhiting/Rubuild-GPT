"""
说明：本脚本实现了一个基于字符级 bigram 的语言模型，用 PyTorch 训练并生成文本。
            该模型学习在给定一个字符的情况下，预测下一个字符的概率分布（本质上是一个 softmax over vocabulary 的简单嵌入层）。
            
主要功能：
1. 读取并处理原始文本数据（input.txt），构建字符级词表；
2. 构建并训练一个简单的 bigram 神经网络语言模型；
3. 每隔一定轮数记录训练集与验证集的平均损失；
4. 在训练结束后，基于训练好的模型生成一定长度的随机文本；
5. 将训练损失和生成文本保存到日志文件 loss_log.txt 以便可视化分析。

输出：
- 日志文件 loss_log.txt，记录了训练过程中的损失变化以及生成的字符序列
"""

import torch
import torch.nn as nn
from torch.nn import functional as F 

# hyperparameters
batch_size = 32
block_size = 8 # what is the maximum context length for prediction?
max_iter = 5000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

# --------------------------
# 文本预处理
# -------------------------
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# 构建字符级词表 （去重+排序）
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 构建字符到索引和索引到字符的映射
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])

# 整个文本编码为整数
data =torch.tensor(encode(text),dtype=torch.long)

# 划分训练集和验证集（90% / 10%）
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# --------------------------
# 构建数据批次
# --------------------------
def get_batch(split):
    # generate a small batch of data of inputs x and tagrgets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    x, y = x.to(device),y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    """在训练集和验证集上评估平均损失"""
    out={}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --------------------------
# 定义 Bigram 语言模型
# --------------------------
class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        # 令每个 token 的嵌入维度等于词表大小（即用 one-hot 风格表示）
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx) #(B,T,C)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        # 基于当前输入 idx（形状 (B, T)）逐步生成新 token
        for _ in range(max_new_tokens):
            logits,loss = self(idx) # get the predictions
            logits = logits[:,-1,:] # focus only on the last time step
            probs = F.softmax(logits,dim=-1) #(B,C) # apply softmax to get probabilities
            idx_next = torch.multinomial(probs,num_samples=1) #(B,1) # sample from the distribution
            idx = torch.cat((idx,idx_next),dim=1) # append sampled index to the running sequence
        return idx

# --------------------------
# 模型训练主流程
# --------------------------
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

# open log file
log_file = open("bigram_loss_log.txt","w")

for iter in range(max_iter):
    
    # every once in a while evaluate the loss on the train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        log_line = f"step {iter}:train loss{losses['train']:.4f},val loss{losses['val']:.4f}"
        print(log_line)
        log_file.write(log_line+"\n")
        
    # sample a batch of data
    xb,yb = get_batch('train')
    
    # evaluate the loss and renew the parameters
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# --------------------------
# 文本生成并记录
# --------------------------
context = torch.zeros((1,1),dtype=torch.long,device=device)
generated_indices = m.generate(context,max_new_tokens=500)[0].tolist()
generated_text = decode(generated_indices)

print("\nGenerated text:")
print(generated_text)

log_file.write("\nGenerated text:\n")
log_file.write(generated_text+"\n")

# close log file
log_file.close()
