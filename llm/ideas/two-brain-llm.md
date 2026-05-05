# Two Brain llm

模仿左右脑结构，训练时候有两个完整的decoder-only的模型
使用中间层张量传输来通信，最后用其中一个来作为最终输出

## 架构示意图

```mermaid
graph TB
    Input["输入 Token"]
    
    subgraph Left["左脑 Decoder Model"]
        LD1["Layer 1"]
        LD2["..."]
        LD3["Layer n"]
    end
    
    subgraph Right["右脑 Decoder Model"]
        RD1["Layer 1"]
        RD2["..."]
        RD3["Layer n"]
    end
    
    Input --> LD1
    Input --> RD1
    
    LD1 --> LD2
    LD2 --> LD3
    RD1 --> RD2
    RD2 --> RD3
    
    LD2 -.中间层张量.-> RD2
    RD2 -.中间层张量.-> LD2
    
    LD3 --> OutputL["左脑输出"]
    RD3 --> OutputR["右脑输出"]
    
    OutputL --> Fusion["融合层<br/>Fusion Layer"]
    OutputR --> Fusion
    
    Fusion --> Output["最终预测"]
```

## 评价

### 优点
- 多路径推理，模型可学到互补特征
- 冗余机制提高稳定性
- 中间层互动允许"协商"

### 核心问题
- **计算成本翻倍** — 同样成本下不如训练更大的单一模型
- **收敛稳定性** — 两个模型梯度相互影响
- ~~**输出设计过简**~~ **→ 改为融合层** — 用可学习的融合机制组合两个输出

### 改进方案
用轻量级融合层代替简单选择：
- 学习加权组合（learned weights）
- 或使用 attention 机制决定融合比例
- 这样两个"脑"的输出都能对最终决策有贡献

