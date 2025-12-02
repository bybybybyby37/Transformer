# models/sampler.py
import numpy as np
from torch.utils.data import Sampler

class TokenBucketSampler(Sampler):
    def __init__(self, dataset, max_tokens=2000, shuffle=True):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.lengths = np.array(dataset.lengths)  
        self.indices = np.arange(len(self.lengths))

    def __iter__(self):
        # sort the index by LENGTH
        sorted_indices = self.indices[np.argsort(self.lengths[self.indices])]

        # generate Batches
        batches = []
        curr_batch = []
        curr_tokens = 0
        
        max_len_in_batch = 0

        for idx in sorted_indices:
            l = self.lengths[idx]
            
            # cauculate the token size and prevent Overflow
            new_max_len = max(max_len_in_batch, l)
            if (len(curr_batch) + 1) * new_max_len > self.max_tokens:
                if curr_batch:
                    batches.append(curr_batch)
                curr_batch = [idx]
                curr_tokens = l
                max_len_in_batch = l
            else:
                curr_batch.append(idx)
                curr_tokens += l
                max_len_in_batch = new_max_len
        
        if curr_batch:
            batches.append(curr_batch)

        # Shuffle the order of the Batches
        if self.shuffle:
            np.random.shuffle(batches)

        # Yield
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.dataset) // (self.max_tokens // 20)