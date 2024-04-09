'''
Mixed-batch data loading and processing logic.
Created by Basile Van Hoorick, Sep 2022.
'''

from __init__ import *


class MixedDataset(torch.utils.data.Dataset):
    '''
    Combines multiple dataset classes into a single instance that interleaves examples from each
        dataset with strictly equal proportions.
    '''

    def __init__(self, logger, dset_dict, chunk_size=1, size_mode='max'):
        '''
        dset_dict (dict): Already instantiated datasets.
        chunk_size (int): Typically batch size to avoid literally mixing examples within batches.
        size_mode (str): ari_mean / geo_mean / max.
        '''
        self.logger = logger
        self.dset_dict = dset_dict
        self.chunk_size = chunk_size
        self.size_mode = size_mode
        self.source_names = list(dset_dict.keys())
        self.num_sources = len(self.source_names)
        self.source_sizes = {k: len(v) for (k, v) in dset_dict.items()}

        if self.size_mode == 'ari_mean':
            # Combined size of this dataset is the sum of all sources.
            self.mixed_size = int(sum(self.source_sizes.values()))

        elif self.size_mode == 'geo_mean':
            # Combined size of this dataset is the geometric mean of all sources, times the number
            # of sources.
            log_sizes = np.log(list(self.source_sizes.values()))
            self.mixed_size = int(np.exp(np.mean(log_sizes))) * self.num_sources

        elif self.size_mode == 'max':
            # Combined size of this dataset is the maximum among all sources.
            self.mixed_size = int(max(self.source_sizes.values()))

        else:
            raise ValueError('Unknown size mode: {}'.format(self.size_mode))

        # Compute the number of chunks per dataset.
        self.num_chunks_per_dset = dict()
        for dset_name, dset in self.dset_dict.items():
            self.num_chunks_per_dset[dset_name] = int(np.ceil(len(dset) / self.chunk_size))

        # Compute the number of chunks per dataset.
        self.num_chunks = 0
        for dset_name, dset in self.dset_dict.items():
            self.num_chunks += self.num_chunks_per_dset[dset_name]

    def __len__(self):
        return int(self.mixed_size)

    def __getitem__(self, index):
        '''
        Interleaves examples from different dataset sources (which may each have unique sizes),
            while interpreting the chunk size as the allowed frequency of flipping. For example, we
            will return these example indices from these sources:
                 0 1 2 3    0 1 2 3   4 5 6 7    4 5 6 7
                [ kubric ] [ davis ] [ kubric ] [ davis ]
            Once any dataset class runs out of examples (e.g. its size is less than our size), then
            we simply wrap around back to index 0 for that source.
            NOTE: The dataset instances themselves must support shuffling to avoid imbalanced
            sampling.
        '''
        chunk_idx = (index // self.chunk_size) // self.num_sources
        src_idx = (index // self.chunk_size) % self.num_sources
        src_name = self.source_names[src_idx]
        src_size = self.source_sizes[src_name]
        within_idx = (chunk_idx * self.chunk_size + (index % self.chunk_size)) % src_size

        data_retval = self.dset_dict[src_name][within_idx]

        mixed_info = dict()
        mixed_info['iter_idx'] = index
        mixed_info['chunk_idx'] = chunk_idx
        mixed_info['src_name'] = src_name
        mixed_info['src_size'] = src_size
        mixed_info['within_idx'] = src_idx
        data_retval['mixed_info'] = mixed_info  # Assumes data_retval is always a dict itself.

        # DEBUG:
        self.logger.debug(f'mixed_info: {mixed_info}')

        return data_retval
