import torch
import torch.nn as nn
from torch.nn import functional as F 

# hyperparameters
batch_size = 32
block_size = 8 # what is the maximum context length for prediction?
max_iter = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

# calculate all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s:[stoi[c] for c in s]
decode = lambda l:''.join([itos[i] for i in l])

# train and test split
data =torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
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

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # h(x): (B,T,head_size)

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1) # (B,T,num_heads * head_size)

class Head(nn.Module):
    """ one head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self,x):
        B,T,V = x.shape
        k = self.key(x)
        q = self.query(x)
        v  =self.value(x)
        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,16)@(B,16,T) ----> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) 
        wei = F.softmax(wei,dim=-1)
        # perform the weighted aggregation of the values
        v  =self.value(x) #(B,T,C), C:n_embd
        out = wei@v # (B T T) @ (B T C)----> (B T C)
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,n_embd),
            nn.ReLU(),
        )
    
    def forward(self,x):
        return self.net(x)

# super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.sa_head = MultiHeadAttention(4,n_embd//4) # 4 heads of 8-dimensional self-attention 
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        
    def forward(self,idx,targets=None):
        B,T = idx.shape
        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) #(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # (T,C)
        x = tok_emb+pos_emb #(B T C)
        x = self.sa_head(x) # (B T C)
        x = self.ffwd(x)  #(B T C)
        logits = self.lm_head(x) # (B T vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B T) array of indices in the corrent context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]
            # get the predictions
            logits,loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:,-1,:]
            # apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1) #(B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs,num_samples=1) #(B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

log_file = open("gpt_mha_ffwd_loss_log.txt","w")

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
