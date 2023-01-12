# Importing required libaries
import pandas as pd
import numpy as np

import os
import glob
import logging
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer
)


# ---- Classes used for training, etc. ----
class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)

        directory = args.chache_dir
        cached_features_file = os.path.join(
            directory, args.model_type + '_cached_lm_' + str(block_size)
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info('Loading features from cached file %s', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file %s", directory)

            self.examples = []
            for _, row in df.iterrows():
                conv = contruct_conv(row, tokenizer)
                self.examples.append(conv)
            
            logger.info('Saving features into cached file %s', cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

class Args():
    def __init__(self):
        self.output_dir = 'output-small'
        self.model_type = 'gpt2'
        self.model_name_or_path = 'microsoft/DialoGPT-small'
        self.config_name = 'microsoft/DialoGPT-small'
        self.tokenizer_name = 'microsoft/DialoGPT-small'
        self.cache_dir = 'cached'
        self.block_size = 512
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = False
        self.per_gpu_train_batch_size = 4
        self.per_gpu_eval_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 16
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_steps = 3500
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.overwrite_cache = True
        self.should_continue = False
        self.seed = 42
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = 'O1'


# ---- Collecting Data ----

# Getting first csv and modifying to desired form
script1 = pd.read_csv('src/Family_guy_dialog.csv')
del script1['seasons']
script1.rename(columns={'character': 'name'}, inplace=True)
script1.rename(columns={'dialog': 'line'}, inplace=True)
# print(script1.sample(10))

# Getting second csv
script2 = pd.read_csv('src/Family_guy.csv')
# print(script2.sample(10))

context = []
# context window size of 7
n = 7

Character_Name = 'Stewie'

# Collecting data for stewie from script one
for i in script1[script1.name == Character_Name].index:
    if i < n:
        continue

    row = []
    prev = i - 1 - n # Previous context
    
    # Adding context
    for j in range(i, prev, -1):
        row.append(script1.line[j])

    context.append(row)

# Creating panda for stewie in script 1
columns = ['response', 'context']
columns = columns + ['context/' + str(i) for i in range(n - 1)]
stewieScript1 = pd.DataFrame.from_records(context, columns=columns)
# print(len(stewieScript1))
# print(stewieScript1.sample(10))

# Collecting data for stewie from script two
for i in script2[script2.name == Character_Name].index:
    if i < n:
        continue

    row = []
    prev = i - 1 - n # Previous context
    
    # Adding context
    for j in range(i, prev, -1):
        row.append(script2.line[j])

    context.append(row)

# Creating panda for stewie in script 2
columns = ['response', 'context']
columns = columns + ['context/' + str(i) for i in range(n - 1)]
stewieScript2 = pd.DataFrame.from_records(context, columns=columns)
# print(len(stewieScript2))
# print(stewieScript2.sample(10))

stewieScriptFinal = pd.concat([stewieScript1, stewieScript2])

# Creating training and testing data
train_set, test_set = train_test_split(stewieScriptFinal, test_size=0.1)
# print(train_set.head())
# print(test_set.head())

