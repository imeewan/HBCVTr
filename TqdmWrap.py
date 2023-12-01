#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tqdm import tqdm
from tqdm.auto import tqdm


class TqdmWrap:
    def __init__(self, tqdm_instance):
        self.tqdm_instance = tqdm_instance

    def __call__(self, batch):
        self.tqdm_instance.update(1)
        return batch

