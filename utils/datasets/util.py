
import importlib
import random
from copy import deepcopy

import torchvision.transforms as transforms

from utils.options.options import parse_params, parse_arguments


def generate_random_indices_in_a_sequence(sequence_length, count, padding_mode='clip'):
    """
    Generate random indices in a sequence.

    padding_mode == 'clip':
        e.g.
            sequence = [0, 1, 2, 3, 4, 5], count = 5
            return [0, 1, 2, 3, 4] or [1, 2, 3, 4, 5]
    padding_mode == 'replicate':
        e.g.
            sequence = [0, 1, 2, 3, 4, 5], count = 5
            return [0, 0, 0, 1, 2] or [0, 0, 1, 2, 3] or ... or [2, 3, 4, 5, 5] or [3, 4, 5, 5, 5]
    padding_mode == 'reflect':
        e.g.
            sequence = [0, 1, 2, 3, 4, 5], count = 5
            return [2, 1, 0, 1, 2] or [1, 0, 1, 2, 3] or ... or [2, 3, 4, 5, 4] or [3, 4, 5, 4, 3]
    padding_mode == 'bounce':
        e.g.
            sequence = [0, 1, 2, 3, 4, 5], count = 5
            return [4, 3, 0, 1, 2] or [4, 0, 1, 2, 3] or ... or [2, 3, 4, 5, 1] or [3, 4, 5, 2, 1]
    padding_mode == 'circle':
        e.g.
            sequence = [0, 1, 2, 3, 4, 5], count = 5
            return [4, 5, 0, 1, 2] or [5, 0, 1, 2, 3] or ... or [2, 3, 4, 5, 0] or [3, 4, 5, 0, 1]


    Args:
        sequence_length (int): The length of the sequence to generate random indices from.
        count (int): The number of indices to generate.
        padding_mode (str, optional): The padding mode. Defaults to 'clip'.

    Returns:
        list: The generated random indices.
    """
    seq_length = sequence_length
    if padding_mode == 'clip':
        random_range = (0, seq_length - count - 1)
        start = random.randint(*random_range)
        indices = [start + i for i in range(count)]
    elif padding_mode in ['replicate', 'reflect', 'bounce', 'circle']:
        n_max = seq_length - 1
        n_pad = count // 2
        random_range = ((count + 1) % 2, n_max)    # 窗口超出 seq 区域不能大于 count // 2
        start = random.randint(*random_range)
        if padding_mode == 'replicate':
            def pad(x, anchor):
                if x < 0:
                    return 0
                elif x > n_max:
                    return n_max
                else:
                    return x
        elif padding_mode == 'reflect':
            def pad(x, anchor):
                if x < 0:
                    return -x
                elif x > n_max:
                    return 2 * n_max - x
                else:
                    return x
        elif padding_mode == 'bounce':
            def pad(x, anchor):
                if x < 0:
                    return (anchor + n_pad) - x
                elif x > n_max:
                    return (anchor - n_pad) - (x - n_max)
                else:
                    return x
        elif padding_mode == 'circle':
            def pad(x, anchor):
                if x < 0:
                    return x + seq_length
                elif x > n_max:
                    return x - seq_length
                else:
                    return x
        else:
            raise ValueError(f"padding_mode '{padding_mode}' not supported")

        indices = [pad(start + i, anchor=start) for i in range(-n_pad, count - n_pad)]
    else:
        raise ValueError(f"padding_mode '{padding_mode}' not supported")

    return indices
