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
)

# Getting first csv and modifying to desired form
script1 = pd.read_csv('Family_guy_dialog.csv')
del script1['seasons']
script1.rename(columns={'character': 'name'}, inplace=True)
script1.rename(columns={'dialog': 'line'}, inplace=True)
# print(script1.sample(10))

# Getting second csv
script2 = pd.read_csv('Family_guy.csv')
# print(script2.sample(10))

contexted = []
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

    contexted.append(row)

# Creating panda for stewie in script 1
columns = ['response', 'context']
columns = columns + ['context/' + str(i) for i in range(n - 1)]
stewieScript1 = pd.DataFrame.from_records(contexted, columns=columns)
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

    contexted.append(row)

# Creating panda for stewie in script 2
columns = ['response', 'context']
columns = columns + ['context/' + str(i) for i in range(n - 1)]
stewieScript2 = pd.DataFrame.from_records(contexted, columns=columns)
# print(len(stewieScript2))
# print(stewieScript2.sample(10))