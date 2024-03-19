import os
from typing import Union, List

from pathlib import Path
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOKENIZER_DIR = Path(__file__).parent / 'tokenizers'
DEFAULT_TOKENIZER = 'conch_byte_level_bpe_uncased.json'

def get_tokenizer():
    tokenizer = PreTrainedTokenizerFast(tokenizer_file = str(TOKENIZER_DIR / DEFAULT_TOKENIZER), 
                                        bos_token="<start_of_text>",
                                        eos_token="<end_of_text>",
                                        pad_token="<pad>")
    return tokenizer

def tokenize(tokenizer, texts):
    # model context length is 128, but last token is reserved for <cls>
    # so we use 127 and insert <pad> at the end as a temporary placeholder
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = 127,
                                        add_special_tokens=True, 
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_tensors = 'pt')
    tokens = F.pad(tokens['input_ids'], (0, 1), value=tokenizer.pad_token_id)
    return tokens
