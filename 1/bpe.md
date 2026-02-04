byte pair encoding (BPE)
字节对编码最初是为了压缩文本开发

统计字节频率之前还要进行预分词
防止模型学习到跨越语义边界的无意义组合

流程：
构建统计字典，然后寻找最频繁的字节对，进行合并
比如
`('l','o') => ('lo')`

具体来说
比如
lower和low两个单词
分解之后变成
[
  ['l','o','w'],  // Vec<u8>
  ['l','o','w','e','r'] // Vec<u8>
] // Vec<Vec<u8>>
这个就是word_list

先统计词频，比如low出现了5次，lower出现了3次
就是
Map: [
    ['l','o','w'] : 5,
    ['l','o','w','e','r'] : 3
]
这个就是count_list 和word_list是一一对应的

然后要根据word_list，统计所有相邻字节对

```python
for idx,word in enumerate(words_list):
    freq = count_list[idx]
    for i in range(len(word) -1 ):
        pair = (word[i],wordp[i+1])
        stats[pair] += freq
```

为啥要这么算呢，因为很多单词会重复出现，比如low，
我只需要统计lo和ow这两个pair，然后加到pair对统计中，
然后lo的次数就是我的low单词出现的次数，所以统计lo和ow频率的时候，
只需要统计low出现的次数就行了

接下来找到best_pair 以及哪些单词中有这个best_pair进行更新
同时把best_pair 进行合并，此时所有受到影响的单词都会将l和o变为lo
比如['l','o','w'] => ['lo','w']
然后要删掉过去的做邻居和右邻居
比如原来是['a','l','o','w'] 
现在lo变成一个了，之前的pair('a','l')和('o','w')都需要减去当前这个单词的频率
也就是如果alow的频率是10，stat里面('a','l')和('o','w')的频率都需减去10

然后删掉之后就要生成新的pair，针对左邻居和右邻居，
就需要生成新的pair，比如lo后面是w，新的pair就是('lo',w')，
stat里面('lo','w')的freq就需加上这个low单词出现的次数

现在的问题就是，我找到一个best pair之后，如何知道wordlist里面哪些单词受到了影响？
答案就是使用倒排索引
我的每一个pair都保存了一个对应的单词在wordlist和countlist中的下标
也就是使用一个
HashMap<(Vec<u8>,Vec<u8>),HashSet<usize>>
这个在遍历的时候可以直接存起来
```py 
for idx,word in enumerate(words_list):
    freq = count_list[idx]
    for i in range(len(word) -1):
        pair = (word[i],wordp[i+1])
        stats[pair] += freq
        indices.entry(pair).or_insert_with(HashSet::new).insert(idx)
```

这样每次得到一个bestpair之后，可以通过indices[pair]得到受到影响的单词的Set
然后遍历这个Set里面的idx，得到都是哪些单词受到了影响，
然后进行删除旧的pair的freq，增加新的pair的freq

最后统计完成之后，当前的best pair由于已经在所有受到影响的单词中都合并完成了
所以需要删除掉这个best pair，在stats里面删掉best pair，倒排索引中这个也要删掉

完成迭代之后，merge数组里面已经保存了所有的合并pair，这个merge数组就是一个bpe合并rules
接下来就是将这些新的pair添加到vocab里面(id 从256开始，前255是)
然后再把special token 添加到vocab里面
```python
for pair in merges:
    new_id = len(vocab)
    vocab[new_id]=pair[0]+pair[1]

for s_token in special_tokens:
    s_byte = s_token.encode("utf-8")
    vocab[len(vocab)] = s_byte
```

由于一些字符是不可见的，所以需要进行一些转换
以下是gpt2的做法
将原始字节数据（如文本文件的二进制内容）转换为模型可处理的 Unicode 字符序列
由于字节范围是 0-255（共 256 个值），但部分字节（如控制字符）无法直接显示为可见字符
所以需要将常用的ascii可打印字符保留，映射为自身
那些不可见的字符就需要映射到256以上的unicode码点，确保所有字符可见
```python
bs = list(range(ord("!"), ord("~") + 1))  # ASCII 可打印字符 33-126（! 到 ~）
       + list(range(ord("“"), ord("-") + 1))  # 中文引号等字符（具体范围需结合编码）
       + list(range(ord("©"), ord("ÿ") + 1))  # 版权符号 © 到 ÿ（169-255）
cs = bs[:]  # 初始时，cs 与 bs 一一对应
for b in range(256):
    if b not in bs:  # 如果字节不在预定义的可见字符范围内
        bs.append(b)      # 将字节加入 bs
        cs.append(256 + n)  # 将字节映射到 256 以上的 Unicode 码点
        n += 1
return dict(zip(bs,cs))
```
映射示例（ai写的）
```
byte   0 → Unicode 256 → 'Ā' (U+0100)
byte   1 → Unicode 257 → 'ā' (U+0101)
byte   2 → Unicode 258 → 'Ă' (U+0102)
byte   3 → Unicode 259 → 'ă' (U+0103)
byte   4 → Unicode 260 → 'Ą' (U+0104)
byte   5 → Unicode 261 → 'ą' (U+0105)
byte   6 → Unicode 262 → 'Ć' (U+0106)
byte   7 → Unicode 263 → 'ć' (U+0107)
byte   8 → Unicode 264 → 'Ĉ' (U+0108)
byte   9 → Unicode 265 → 'ĉ' (U+0109)
byte  10 → Unicode 266 → 'Ċ' (U+010A)
```

接下来根据两个核心产物merges(rules) 和 vocab(id->Token)

encoder_map:接下来需要将token的字节序列映射为id
decoder_map:逆映射，需要将id映射回去字节序列

Bpe Ranks: Map<(bytes,bytes),int> key是一个pair，value是在merge列表中的索引，表示优先级，越小越好

encode过程：
先进行pre-tokenization 使用正则表达式切分文本为基础单元
然后映射为字节
然后对于每个子串进行迭代合并：对于一个子串，找出里面最优的best_pair，然后进行合并处理
，然后将剩下的符号映射为id

decode过程：
将id列表转换为token列表，然后拼接，然后将unicode字符映射回原始的byte值
然后对这个byte数组进行解码，使用error='replace'来处理无效序列

在大模型训练中，我们不会直接读取文本文件，
文本解析慢，Python 对象内存占用大。
所以对整个数据集进行Tokenize，转换为uint16的紧致的二进制文件`.bin`

