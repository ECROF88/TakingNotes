合成数据如何获取

1. self-instruct，自我指令生成
每个数据打上task type，然后准备一些seed prompt，随机采样seed，
喂给强一些的模型，基于这些seed问题再写一些问题
也可以通过改写的方式，合成不同格式的(文本，markdown，json)
2. reject sampling 拒绝采样
让模型对同一个问题生成多个回答，使用一个验证器或者人工挑选出比较好的
Bon-Best of N 随机采样N个答案，使用奖励模型RM对N个答案打分，选出分高的
Won-Wosrt of N 关注分低的那一个，用来构造负样本，比如DPO中需要一个好答案和坏答案
另外如果一个模型WoN都是对的可以把这个题目删掉，因为太简单
3. SandBox验证

数据质量过滤

1. IFD instruction Following Difficulty 指令跟随难度
先从数据集中选一个子集，给模型微调
然后计算两个得分
    - 带指令损失 在给定指令Q的时候生成正确答案A的难度
    - 不带指令损失 
得分越高说明答案生成难度越大
$$
r_\theta\left(Q,A\right)=\frac{s_\theta\left(A\right)}{s_\theta\left(A|Q\right)}
$$

2. MoDs过滤  Model-based Data Selection
- 使用模型对数据进行质量打分
- 使用通K-center-greedy算法进行数据筛选，都得最大化多样性的轻快俩下使得指令数据最小
- 进行必要性筛选
    - 如果种子模型在一个数据上表现已经很好了，说明这个数据没用了，剩下的就是必要性数据集，然后再对必要性数据集进行多样性筛选获取一个增强数据集

数据多样性
1. prompt 表达方式多样性
2. prompt 难度暴恐，比如Wizard方法进行指令进化
3. prompt 长度均衡
4. answer 长度均衡，不能让模型输出token太少就完事
5. 多轮聊天切换topic能力，有时候query和当前session无关
6. answe 分布多样性，不能让模型过拟合
7. 数据用途的多样性，例如emoji，文本，以及一些下游任务特殊需求
8. 语义级别的多样性
    - topN采样，线性采样，对数采样
    - token词频，向量空间距离
    - 数据embedding，利用k-means和KNN聚类获得数据多样性权重

1. 不同的task type要有不同的channel loss
2. special token一开始loss肯定是有点高
3. sft阶段不能用packing，有损泛化能力
packing是数据打包的意思，因为在大模型训练中，GPU 的显存和计算资源是按固定长度（比如 max_seq_len=4096）分配的。
如果sft数据只有500，剩下的空间会填上`[PAD]`占位符。
Do We Reallly Need Packing in LLM SFT 

训练策略：
1. 多任务学习直接混合不同sft数据越进行sft
2. 使用顺序训练，在每个数据集上依次SFT
3. 先在专业数据上面进行多任务学习，再在通用数据集上面应用SFT
4. 双阶段混合微调

多轮对话提升：
1. 加入伪多轮，就是那种一个session中有转折点的
2. 可以合成一些多轮数据
3. 多轮合并加速计算，可以将3轮样本合并为一个样本。
    但是缺点是因为计算CE的时候，使用reduction为mean时候，loss会使得输出较短的权重降低了所以短输出的数据训练不充分
    所以需要自己设计损失函数
