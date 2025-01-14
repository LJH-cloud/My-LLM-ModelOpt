import torch
from colorama import Fore, init

def slice1d(x, start, end):
    return x[:, start:end, ...]

class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        notice=True
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = slice1d
        self.v_slice = slice1d

        self.length = 0
        self.length_before_generate = 0

        self.notice = notice

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            self.length_before_generate = seq_len
            return past_key_values
        evicted_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        self.length_before_generate = evicted_key_values[0][0].size(self.k_seq_dim)
        if self.notice:
            init(autoreset=True)
            print(Fore.RED + f'KV Cache has evicted for space: {evicted_key_values[0][0].size()}, seq_len: {seq_len}, recent_size: {self.recent_size}, num_coming: {num_coming}, equal: {all([torch.equal(past_key_value, evicted_key_value) for past_key_value, evicted_key_value in zip(past_key_values[0], evicted_key_values[0])])}')
        return evicted_key_values

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def update_kv_length(self, past_key_values_in):
        length_after_generate = past_key_values_in[0][0].size(self.k_seq_dim)
        self.length += length_after_generate - self.length_before_generate
        if self.notice:
            init(autoreset=True)
            print(Fore.RED + f'before generate kv cache size: {self.length_before_generate}, after generate kv cache size: {self.length}')
