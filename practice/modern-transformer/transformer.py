import torch
import math
import einx
from torch import nn
from torch import Tensor
from jaxtyping import Float
from collections.abc import Iterable

class Linear(nn.Modeule):
    def __init__(self,in_features,out_features,weights:Float[Tensor," out in"]|None=None,device=None,dtype=None):
        super().__init__()
        if weights is None:
            # 使用截断正态分布初始化权重
            # 保证向前传播和向后传播梯度的方差稳定
            sigma=math.sqrt(2/(in_features+out_features))
            self.w=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
            nn.init.trunc_normal_(self.w,mean=0.0,std=sigma,a=-3*sigma,b=3*sigma)
        else:
            self.w=nn.Parameter(weights)
    # 向前传播，使用einx进行爱因斯坦求和约定
    # ... 表示任意数量的批次维度
    # [in] 表示输入特征维度
    # out [in] 表示权重矩阵的形状
    # 结果形状为 ... out
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return einx.dot("... [in], out [in] -> ... out", x, self.w)

# 把离散token直接映射到向量空间。
class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        self.embeddings=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        # 使用截断正态分布初始化权重
        # 标准差 1/sqrt(embedding_dim)，有利于稳定训练
        nn.init.trunc_normal_(self.embeddings,mean=0.0,std=1/math.sqrt(embedding_dim))
    # 向前传播
    # 输入: token_ids（通常形状 [batch, seq] 或更高维）
    # 输出: 利用张量索引取行，得到对应的向量，输出形状为 [*token_ids.shape, embedding_dim]。
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        return self.embeddings[token_ids]

# 归一化，但RmsNorm不减平均值
class RmsNorm(nn.Module):
    def __init__(self,d_model:int,eps=1e-5,device=None,dtype=None):
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 转换输入类型，提升数值稳定性
        input_dtype=x.dtype
        x=x.to(torch.float32)
        # 计算方差
        variance=x.pow(2).mean(dim=-1,keepdim=True)
        # 归一化
        x=x*torch.rsqrt(variance+self.eps)
        # 缩放
        return (self.g*x).to(input_dtype)

# 激活函数
class SiLu(torch.nn.Module):
    def _init_(self):
        super().__init__()
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x*torch.sigmoid(x)

# 门控线性单元，作用是控制信息流，使模型更加稳定
class Glu(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        # w1是门控信号，用于控制信息流
        # w2是线性变换，用于将信息流转换为输出
        self.w1=Linear(in_features,out_features,device=device,dtype=dtype
        )
        self.w2=Linear(in_features,out_features,device=device,dtype=dtype)
    # 向前传播
    # 输入: x（通常形状 [batch, seq, in_features]）
    # 输出: 门控信号与线性变换的乘积，输出形状为 [batch, seq, out_features]。
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return torch.sigmoid(self.w1(x))*self.w2(x)

# 门控线性单元，SiLU 的平滑梯度与门控抑制极值有助于深层稳定。
class SwiGlu(nn.Module):
    def __init__(self,d_in:int,d_hidden:int,d_out:int,device=None,dtype=None)->None:
        super().__init__()
        # w1是门控信号，用于控制信息流
        self.w1=Linear(d_in,d_hidden,device=device,dtype=dtype)
        # w2回到输出维度
        self.w2=Linear(d_hidden,d_out,device=device,dtype=dtype)
        # w3提供值
        self.w3=Linear(d_in,d_out,device=device,dtype=dtype)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.w1(x)*self.w2(x)+self.w3(x)

FFN = SwiGlu

# 旋转位置编码
class RoPE(nn.Module):
    def __init__(self, dim:int,max_seq_len:int=2048,theta:float=10000,device=None,dtype=None):
        super().__init__()
        self.dim=dim
        self.max_seq_len=max_seq_len

        # inv_freq: (dim//2,) 频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

        # t: (seq_len,) 时间
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # freqs: (seq_len, dim//2) 频率
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # outer product

        emb = freqs.repeat_interleave(2, dim=-1)  # (seq_len, dim) 嵌入

        # Now register buffers
        self.register_buffer("cos_cached", emb.cos().to(dtype))  # (seq_len, dim) 余弦
        self.register_buffer("sin_cached", emb.sin().to(dtype))  # (seq_len, dim) 正弦

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Float[Tensor, "... seq"]) -> torch.Tensor:
        # token_positions: (..., seq_len) 任意前缀维度
        # x: (..., seq_len, dim)
        cos = self.cos_cached[token_positions]  # (..., seq_len, dim)
        sin = self.sin_cached[token_positions]  # (..., seq_len, dim)
        # 这一步是拆分，把dim//2个特征拆分成两个特征，方便后面旋转
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # (..., seq_len, dim//2, 2)
        # 旋转：(a,b) -> (-b,a)
        x_rotated = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)  # rotate: (a,b) -> (-b,a)
        # 恢复形状
        x_rotated = x_rotated.view(*x.shape)  # (..., seq_len, dim)
        # 扩展维度
        # 如果x是4维，则扩展维度
        if x.ndim == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # 这一步是旋转，但实际是混合两个特征，sin是偶数位置，cos是奇数位置
        x_rot = x * cos + x_rotated * sin
        return x_rot

# softmax层，把实数转换为概率分布
class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x:torch.Tensor,dim=1)->torch.Tensor:
        max_x=torch.max(x,dim=dim,keepdim=True).values
        x=x-max_x
        exp_x=torch.exp(x)
        return exp_x/torch.sum(exp_x,dim=dim,keepdim=True)

# scaledDotProductAttention层，计算qkv点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax=Softmax()
    
    def forward(
        self,
        # 定义query,key,value的形状
        q:Float[Tensor,"... s d"],
        k:Float[Tensor,"... s d"],
        v:Float[Tensor,"... s d"],
        mask:torch.Tensor|None=None,
    )->torch.Tensor:
        d_model=q.shape[-1]
        # 计算qk点积
        att=einx.dot("... s_q [d], ... s_k [d] -> ... s_q s_k",q,k)
        att_scale=att/math.sqrt(d_model)
        # 如果mask不为空，则填充-inf，避免注意力机制关注到无效位置
        if mask is not None:
            if mask.ndim<att_scale.ndim:
                mask=mask.reshape((1,)*(att_scale.dnim-mask.ndim)+mask.shape)

            att_scale=att_scale.masked_fill(mask,-torch.inf)

        # 计算注意力分数
        att_score=self.softmax(att_scale)
        # 计算注意力分数与value的点积
        return einx.dot("... s_q [s], ... [s] d -> ... s_q d", att_score, v)

# 多头注意力机制
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_head: int, max_seq_len=2048, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.out_linear = Linear(d_model, d_model, device=device, dtype=dtype)
        self.project = Linear(in_features=d_model, out_features=3 * d_model, device=device, dtype=dtype)
        self.dot_product_att = ScaledDotProductAttention()

        # Cache causal mask - removed the ~ operator
        # 创建因果掩码，用于避免注意力机制关注到无效位置
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: Float[Tensor, "b s d"]) -> torch.Tensor:
        # 获取序列长度
        seq_len = x.shape[1]

        mask = self.causal_mask[:seq_len, :seq_len]
        # 将x投影到q,k,v
        qkv = self.project(x)
        # 将qkv拆分成q,k,v
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)
        # 计算注意力分数
        output = self.dot_product_att(q, k, v, mask)
        # 将output重新排列为b s (h d)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        # 将output线性变换到d_model维度
        return self.out_linear(output)


class MultiHeadAttentionWithRoPE(MultiHeadAttention):
    def __init__(self, d_model: int, num_head: int, theta: float = 10000, max_seq_len=2048, device=None, dtype=None):
        super().__init__(d_model=d_model, num_head=num_head, max_seq_len=max_seq_len, device=device, dtype=dtype)
        self.rope = RoPE(d_model // num_head, max_seq_len=max_seq_len, theta=theta, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[1]
        batch_size = x.shape[0]

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        mask = self.causal_mask[:seq_len, :seq_len]
        # 将x投影到q,k,v
        qkv = self.project(x)
        q, k, v = einx.rearrange("b s (n h d) -> n b h s d", qkv, n=3, h=self.num_head)
        # 将q,k,v应用RoPE
        # Apply RoPE to q and k
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)
        # 计算注意力分数
        output = self.dot_product_att(q, k, v, mask)
        output = einx.rearrange("b h s d -> b s (h d)", output)
        # 将output线性变换到d_model维度
        return self.out_linear(output)

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10000,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        # 归一化
        self.rms_norm1 = RmsNorm(d_model, device=device, dtype=dtype)
        self.rms_norm2 = RmsNorm(d_model, device=device, dtype=dtype)
        # 多头注意力机制
        self.mult_head_atten = MultiHeadAttentionWithRoPE(
            d_model, num_heads, theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        # 前馈神经网络
        self.ffe = FFN(d_model, d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 归一化
        x_norm = self.rms_norm1(x)
        # 多头注意力机制
        x_atten = self.mult_head_atten(x_norm, token_positions)
        x = x + x_atten
        # 归一化
        x_norm = self.rms_norm2(x)
        # 前馈神经网络
        x_ffe = self.ffe(x_norm)
        return x + x_ffe


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        num_layers: int,
        max_seq_len=2048,
        rope_theta: float = 10000,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len

    def forward(self, token_ids: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 词嵌入
        x = self.embedding(token_ids)
        # 如果token_positions为空，则创建token_positions
        if token_positions is None:
            batch_size, seq_len = token_ids.shape
            token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        # 遍历所有transformer块
        for block in self.blocks:
            x = block(x, token_positions)
        # 归一化
        x_norm = self.norm(x)
        # 输出线性变换
        logits = self.out_linear(x_norm)
        return logits


# 梯度裁剪
def gradient_clip(params: Iterable[torch.nn.Parameter], max_norm: float, delta=1e-6):
    with torch.no_grad():
        grads = [p.grad for p in params if p.grad is not None]
        total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach()) for g in grads]))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + delta)
            for g in grads:
                g.detach().mul_(clip_coef)

