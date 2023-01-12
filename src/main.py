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

# create dataset suitable for our model
def contruct_conv(row, tokenizer, oes = True):
    flatten = lambda l: [item for sublist in 1 for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

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


# Cacheing and storing the checkpoints
def load_and_cache_examples(args, tokenizer, df_trn, df_val, evaluate=False):
    return ConversationDataset(tokenizer, args, df_val if evaluate else df_trn)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

# ---- Building the Model ----

tokenier = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')