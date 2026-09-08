---
sr-due: 2026-09-10
sr-interval: 2
sr-ease: 230
---
#code 

# BPE分词算法

世界上有100多万个英文单词，但是只有26个英文字母。而单词的含义确定，单个字母没有含义。我们希望token能准确表示含义，但是又不希望数量太多。
所以我们折中考虑，从单个字母开始合并，在达到想要的数量时停止。
如果语料库很短，我们可以把单词直接作为token，而数量却不多。这时没有必要合并两个单词，因为单词已经是一个独立的个体，合并之后反而含义更加模糊。


## 创建分词表，并进行合并

### build_vocab
>把语料库分成单个字符，并且词尾加上 `</w>`

```python
def build_vocab(corpus):
    tokens = [" ".join(word) + " </w>" for word in corpus.split()]
    # Count frequency of tokens in corpus
    vocab = Counter(tokens)  
    return vocab
```

**具体示例：**
```python
corpus = "low lower low"
build_vocab(corpus)
# 返回 Counter({'l o w </w>': 2, 'l o w e r </w>': 1})
```


### get_stats
>获取pair及其频率

```python
def get_stats(vocab):
    pairs = Counter()
    for word, frequency in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency
    return pairs
```

**具体示例：**
```python
vocab = Counter({'l o w </w>': 2, 'l o w e r </w>': 1})
get_stats(vocab)
# 返回 Counter({('l', 'o'): 3, ('o', 'w'): 3, ('w', '</w>'): 2, ('w', 'e'): 1, ('e', 'r'): 1, ('r', '</w>'): 1})

# 已经合并过的字符不会再分开，如：
# 对于 'lo w </w>'，只能组成 ('lo', 'w') 和 ('w', '</w>')
```

### 合并pair
>把当前词表中「由空格分隔的两个相邻符号」`pair` **合并成一个新符号**，并返回新的词表。

```python
def merge_vocab(pair, v_in):
    v_out = Counter()
    bigram = re.escape(' '.join(pair)) # 防止pair中有转义字符
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # 表示左右两侧是空白或者字符串边界
    for word in v_in: # 对每个词都进行合并
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word] # 频率不变地写入新词表
    return v_out
```

**参数说明：**
- `pair`：一个二元组，例如 `('e', 's')`，表示要把符号 `e` 和 `s` 合并成新符号 `es`。
- `v_in`：当前词表，是一个 `Counter`。key 是**空格分隔的符号串**（如 `'h e s </w>'`），value 是该词的**频率**。

**示例：**
```python
v_in = Counter({
    'h e s </w>': 2,   # "hes"，三个符号还都没合并
    'he s </w>': 1,    # "hes"，但 he 已经是合并过的一个符号
})
merge_vocab(('e', 's'), v_in)
# Counter({'h es </w>': 2, 'he s </w>': 1})
```

### 合并总流程
>先分割成单个字符，每次迭代合并频率最大的pair

```python
with open('temp/HarryPotter.txt', 'r') as f:
    corpus = f.read()

vocab = build_vocab(corpus)
num_merges = 4000 # 合并次数
for i in tqdm(range(num_merges)):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get) # 取频率最大的pair
    vocab = merge_vocab(best, vocab) # 合并单个pair
```

**具体示例（以 `corpus = "low low lower"` 为例）：**

初始词表：`Counter({'l o w </w>': 2, 'l o w e r </w>': 1})`

| 轮次  | 当前词表                                | 频率最大的 pair             | 合并后词表                             |
| --- | ----------------------------------- | ---------------------- | --------------------------------- |
| 1   | `Counter({'l o w </w>': 2, 'l o w e r </w>': 1})` | `('l','o')` = 3        | `Counter({'lo w </w>': 2, 'lo w e r </w>': 1})` |
| 2   | `Counter({'lo w </w>': 2, 'lo w e r </w>': 1})`   | `('lo','w')` = 3       | `Counter({'low </w>': 2, 'low e r </w>': 1})`   |
| 3   | `Counter({'low </w>': 2, 'low e r </w>': 1})`     | `('low','</w>')` = 2   | `Counter({'low</w>': 2, 'low e r </w>': 1})`    |
| 4   | `Counter({'low</w>': 2, 'low e r </w>': 1})`      | `('low','e')` = 1      | `Counter({'low</w>': 2, 'lowe r </w>': 1})`     |
| 5   | `Counter({'low</w>': 2, 'lowe r </w>': 1})`       | `('lowe','r')` = 1     | `Counter({'low</w>': 2, 'lower </w>': 1})`      |
| 6   | `Counter({'low</w>': 2, 'lower </w>': 1})`        | `('lower','</w>')` = 1 | `Counter({'low</w>': 2, 'lower</w>': 1})`       |
| 7   | `Counter({'low</w>': 2, 'lower</w>': 1})`         | 无（`pairs` 为空）→ 跳出循环    | —                                 |

最终词表：`Counter({'low</w>': 2, 'lower</w>': 1})`，即 BPE 学到的子词为 `low</w>` 和 `lower</w>`。

**注意：** pair只能来自一个词内部，合并永远是单词范围的。不能把两个词看作一个token。

---

## 获取tokens 并创建对应id

### get_tokens_from_vocab
>从合并后的vocab中获取token

```python
def get_tokens_from_vocab(vocab):
    tokens_frequencies = Counter()
    vocab_tokenization = {}
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens_frequencies[token] += freq
        vocab_tokenization[''.join(word_tokens)] = word_tokens
    return tokens_frequencies, vocab_tokenization
```


### get sorted_tokens
>对tokens按长度排序，同长度按频率排序

```python
def measure_token_length(token):
    return len(token[:-4]) + 1 if token[-4:] == '</w>' else len(token) # '</w>'看成一个字符
    
sorted_tokens = [
    token
    for (token, freq) in sorted(
        tokens_frequencies.items(),
        key=lambda item: (measure_token_length(item[0]), item[1]),
        reverse=True,
    )
]
```


### 创建id
>对每个token创建唯一id，并且加上两个特殊token：
- `UNK_TOKEN = '</u>'`：没见过的token
- `PAD_TOKEN = '</p>'`：填充用的token

```python
UNK_TOKEN = '</u>'
PAD_TOKEN = '</p>'

token2id = { UNK_TOKEN: 0, PAD_TOKEN:1}
for i, token in enumerate(sorted_tokens):
    token2id[token] = i+2

id2token = {v:k for k, v in token2id.items()}

vocab_size = len(token2id) # 词表中token总数
```

---

## Tokenizer
>获取 `sorted_tokens` 和 `input_id_map` 后，便可以对新输入进行分词

```python
def tokenize(string, sorted_tokens, input_id_map, return_strings=False, max_length=32, unknown_token='</u>'):
    
    string = " ".join([s+'</w>' for s in string.split(" ")])
    
    def tokenize_words(string, sorted_tokens, unknown_token):
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token)

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += tokenize_words(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += tokenize_words(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            break
        return string_tokens
    
    tokens = tokenize_words(string, sorted_tokens, unknown_token)
    if return_strings: return tokens
    if max_length == -1: return [input_id_map[t] for t in tokens]
    input_ids = [1] * max_length
    attention_mask = [0] * max_length
    for i, t in enumerate(tokens):
        input_ids[i] = input_id_map[t]
        attention_mask[i] = 1
    return input_ids, attention_mask
```

**参数说明：**

1. 必传参数（没有默认值）
	- `string`：**待分词的原始文本**，如 `"low lower"`。函数内部会先给它加上 `</w>` 词尾标记，再切分。
	- `sorted_tokens`：**BPE 训练得到的词表**，关键前提：必须按 token 长度**从长到短排序**。因为内部算法是贪心最长匹配，顺序错了结果就错。
	- `input_id_map`：**token → id 的映射表**，如 `{"low</w>": 42, ...}`。只有当返回 id 序列（而非字符串）时才会用到。

2. 可选参数（有默认值）
	- `return_strings`：**控制返回格式**。`True` → 返回子词字符串列表（如 `["low</w>", "er</w>"]`），便于人看；`False` → 返回数值化的 id 序列，便于喂给模型。
	- `max_length`：**序列固定长度**（seq_len）。分词结果会被填充（pad）到这个长度。特殊值 `-1` 表示不填充，直接返回不定长的 id 列表。
	- `unknown_token`：**未知 token 的占位符**。当某段文本在词表中完全找不到匹配时，用它代替（类似 BERT 的 `[UNK]`）。注意它也需要存在于 `input_id_map` 中，否则转 id 时会 `KeyError`。

**示例：**

```python
max_len = 64

input_text = 'I am the first king of england'

tokenized = tokenize(input_text,
                     sorted_tokens=sorted_tokens,
                     input_id_map=token2id,
                     return_strings=True,
                     max_length=max_len,
                     unknown_token='</u>')
print(tokenized)
# ['I</w>', 'am</w>', 'the</w>', 'first</w>', 'king</w>', 'of</w>', 'eng', 'land</w>']

input_ids, attention_mask = tokenize(input_text,
                     sorted_tokens=sorted_tokens,
                     input_id_map=token2id,
                     return_strings=False,
                     max_length=max_len,
                     unknown_token='</u>')
print(input_ids)
print(attention_mask)
# [3418, 2715, 1876, 623, 1149, 2686, 3371, 1266, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 以上数组的长度为64
```

---

## 词嵌入
>子词编号(如 `eng → 107`)是人为按频率等规则分配的，编号大小本身隐含了“频率高低”这类无关信息，这种先验关系对模型反而是噪声，所以要转换成随机初始化的向量，让模型自己学出语义。

```python
emb_dim = 32

# 在N(0, 1)的范围生成随机向量
token_embeddings = nn.Embedding(vocab_size, emb_dim) # vocab_size x embedding_dim

with torch.no_grad():
    input_embeddings = token_embeddings(torch.Tensor(input_ids).long()) # max_len x embedding_dim
# input_ids 与 input_embeddings 的顺序相同，只是内容替换
```