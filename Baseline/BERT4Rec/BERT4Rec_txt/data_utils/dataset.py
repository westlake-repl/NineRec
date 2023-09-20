import torch
from torch.utils.data import Dataset
import numpy as np
import torch.distributed as dist
import math
import random


class BuildTrainDataset(Dataset):
    def __init__(self, u2seq, item_content, item_num, max_seq_len, mask_prob, use_modal):
        self.u2seq = u2seq
        self.item_num = item_num
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        self.use_modal = use_modal
        self.item_content = item_content
        self.MASK_TOKEN = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)
        tokens, positives, negatives, mask_index = [], [], [], []
        for item_id in seq:
            prob = random.random()
            if prob < self.mask_prob:
                tokens.append(self.MASK_TOKEN)
                positives.append(item_id)
                sam_neg = random.randint(1, self.item_num)
                while sam_neg in seq:
                    sam_neg = random.randint(1, self.item_num)
                negatives.append(sam_neg)
                mask_index.append(1)
            else:
                tokens.append(item_id), positives.append(0), negatives.append(0), mask_index.append(0)

        mask_len = self.max_seq_len - seq_Len

        log_mask = [0] * mask_len + [1] * seq_Len
        padding_tokens = [0] * mask_len + tokens
        padding_positives = [0] * mask_len + positives
        padding_negatives = [0] * mask_len + negatives
        padding_mask_index = [0] * mask_len + mask_index

        sample_items = [padding_tokens, padding_positives, padding_negatives]
        sample_items = torch.LongTensor(sample_items).transpose(0, 1)

        if self.use_modal:
            sample_items = self.item_content[sample_items]

        return torch.LongTensor(sample_items), torch.FloatTensor(log_mask), torch.LongTensor(padding_mask_index)


class BuildEvalDataset(Dataset):
    def __init__(self, u2seq, item_content, max_seq_len, item_num):
        self.u2seq = u2seq
        self.item_content = item_content
        self.max_seq_len = max_seq_len
        self.item_num = item_num
        self.MASK_TOKEN = item_num

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, user_id):
        seq = self.u2seq[user_id]
        seq_Len = len(seq)

        tokens = seq[:-1] + [self.MASK_TOKEN]
        target = seq[-1]
        mask_len = self.max_seq_len - seq_Len

        padding_tokens = [0] * mask_len + tokens
        log_mask = [0] * mask_len + [1] * seq_Len

        input_embs = self.item_content[padding_tokens]
        labels = np.zeros(self.item_num)
        labels[target - 1] = 1.0
        return torch.LongTensor([user_id]), \
            input_embs, \
            torch.FloatTensor(log_mask), \
            labels


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
