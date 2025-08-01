from .tokenizers import CharTokenizer
from .tokenizers import SpaceTokenizer
from .tokenizers import BertTokenizer
from .data import *
from .act_fun import *
from .optimizers import *


str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset, "lm": LmDataset, "mlm": MlmDataset,
               "bilm": BilmDataset, "albert": AlbertDataset, "seq2seq": Seq2seqDataset,
               "t5": T5Dataset, "cls": ClsDataset, "prefixlm": PrefixlmDataset}
str2dataloader = {"bert": BertDataLoader, "lm": LmDataLoader, "mlm": MlmDataLoader,
                  "bilm": BilmDataLoader, "albert": AlbertDataLoader, "seq2seq": Seq2seqDataLoader,
                  "t5": T5DataLoader, "cls": ClsDataLoader, "prefixlm": PrefixlmDataLoader}

str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                "polynomial": get_polynomial_decay_schedule_with_warmup,
                "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup}

__all__ = ["CharTokenizer", "SpaceTokenizer", "BertTokenizer", "str2tokenizer",
           "BertDataset", "LmDataset", "MlmDataset", "BilmDataset",
           "AlbertDataset", "Seq2seqDataset", "T5Dataset", "ClsDataset",
           "PrefixlmDataset", "str2dataset",
           "BertDataLoader", "LmDataLoader", "MlmDataLoader", "BilmDataLoader",
           "AlbertDataLoader", "Seq2seqDataLoader", "T5DataLoader", "ClsDataLoader",
           "PrefixlmDataLoader", "str2dataloader",
           "gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
           "AdamW", "Adafactor", "str2optimizer",
           "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
           "get_cosine_with_hard_restarts_schedule_with_warmup",
           "get_polynomial_decay_schedule_with_warmup",
           "get_constant_schedule", "get_constant_schedule_with_warmup", "str2scheduler"]
