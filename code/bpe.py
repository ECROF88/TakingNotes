from typing import Iterable

for pre_token in pre_tokens:
    bytes_part = [bytes([b]) for b in pre_token.encode("utf-8")]
    
    
    
def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
    """
    内存高效的迭代编码器
    分为块处理输入的可迭代对象，并逐个编码每个块
    以减少内存使用
    """
    for chunk in iterable:
        yield from self.encode(chunk) 