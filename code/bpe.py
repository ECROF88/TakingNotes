
def train(
    input_file:str,
    vocab_size:int,
    special_tokens:list[str],
):
    vocab = {i:bytes([i]) for i in range(256)}
    
    # 256 是基础字节，0-1 a-z A-Z 等等
    # vocab_size = 256 + len(special_tokens) + new_tokens_to_add
    num_merges = vocab_size - 256 - len(special_tokens)