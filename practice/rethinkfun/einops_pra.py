import torch
from einops import rearrange, reduce, repeat

# 高维张量示例：4D (b c h w)
x = torch.randn(2, 3, 4, 4)
print('x shape:', x.shape)

# 1) 维度重排：转为 NHWC 格式 (b h w c)
x_nhwc = rearrange(x, 'b c h w -> b h w c')
print('x_nhwc shape:', x_nhwc.shape)

# 2) 归约：对空间维做平均池化，得到 (b c)
x_pooled = reduce(x, 'b c h w -> b c', 'mean')
print('x_pooled shape:', x_pooled.shape)

# 3) 重复：复制 batch 两次，得到 (2*b c h w)
x_repeat = repeat(x, 'b c h w -> (b repeat) c h w', repeat=2)
print('x_repeat shape:', x_repeat.shape)

# 4) ...处理可变数量多轴
x_perm = rearrange(x, '... -> ... a b', a=2, b=3)
print('x_perm shape:', x_perm.shape)