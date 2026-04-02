Bert 仅仅使用Transformer的encoder层
而例如GPT，LLaMA则是decoder-only

decoderonly每个token只能看到之前的信息，不能看到其他后面的信息，
是一种单向的过程

为什么都使用decoder-only：

1. mask是个三角矩阵，而且是满秩的
而encoder的双向注意力会存在低秩问题
2. 训练参数更少，工程实现上有优势
3. Attention矩阵是由一个低秩分解的矩阵加softmax得来
也就是nxd 和 dxn矩阵相乘之后加softmax得到一个nxn矩阵，而n>>d,
这种形式的Attention矩阵会因为低秩问题导致表达能力下降
Decoder only架构的attention是个下三角矩阵，由于softmax导致对角线必然都是正数
而根据线性代数的知识可以知道行列式是对角线之积，所以Decoderonly架构的attention
矩阵必然是满秩的，理论上可以有更强的表达能力
4. 论文实验中的一些研究发现，decoder-only架构下载没有tuning数据的情况下
zero-shot效果更好
5. decoder-only支持复用KV cache，对于多轮对话更加友好。每次新来的q直接乘以k就行了，不需要保存过去的那个q


batch_size :
1. 太大导致了噪音太多，收敛震荡
2. 太小导致了陷入尖锐最小值
