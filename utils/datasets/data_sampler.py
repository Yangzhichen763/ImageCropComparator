import math
import functools
import torch
from torch.utils.data.sampler import Sampler

from utils.options.options import parse_params
from utils.registry import SAMPLER_REGISTRY
from utils.console.log import get_root_logger
logger = get_root_logger()


'''
Adapted from FastLLVE(https://github.com/Wenhao-Li-777/FastLLVE/blob/main/data/data_sampler.py)
'''
@SAMPLER_REGISTRY.register()
class DistIterSampler(Sampler):
    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            seed=0,

            ratio=1,

            **kwargs
    ):
        """
        Distributed Sampler for iterative training.

        Args:
            dataset (Dataset): dataset to sample from
            num_replicas (Optional[int]): number of processes participating in distributed training. Defaults to None.
            rank (Optional[int]): rank of the current process within num_replicas. Defaults to None.
            ratio (int): enlarging ratio. Default: 1.
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.total_size, generator=generator).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        self.epoch += 1

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


"""
e.g. (where [i,j] represents i,i+1,i+2,...,j-2,j-1,j)
(1,2,3,4,5,6,7),8,9,10,11,12          -> [1,7]
1,2,(3,4,5,6,7,8,9),10,11,12,13       -> [3,9]
1,(2,3,4,5,6,7,8),9,10,11,12,13,14    -> [2,8]
1,2,3,(4,5,6,7,8,9,10)                -> [4,10]
1,2,(3,4,5,6,7,8,9),10,11,12          -> [3,9]
shuffle
"""
@SAMPLER_REGISTRY.register()
class VideoClipSampler(Sampler):
    def __init__(
            self,
            video_dataset,
            num_replicas=1,
            rank=0,
            seed=0,

            seq_length=None,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            has_end_signal=False,
            dynamic_batch_size=False,
            batch_first=False,
            end_on_each_frame=False,

            **kwargs
    ):
        """
        Distributed Sampler for video clip training.

        Args:
            video_dataset: a VideoDataset object with frame_counts and video_count attribute.
            seq_length (int): length of each clip.
            batch_size (int): batch size.
            shuffle (bool): whether to shuffle the video order.
            drop_last (bool): whether to drop the last incomplete batch.
            has_end_signal (bool): whether to include end signal in the batch.
            dynamic_batch_size (bool): whether to return a frame-level batch or clip-level batch.
            batch_first (bool): should the first dimension be batch or clip.

            num_replicas (Optional[int]): number of processes participating in distributed training. Defaults to None.
            rank (Optional[int]): rank of the current process within num_replicas. Defaults to None.
            seed (int): random seed.
        """
        super().__init__(video_dataset)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed

        self.video_dataset = video_dataset
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.has_end_signal = has_end_signal
        self.dynamic_batch_size = dynamic_batch_size
        self.batch_first = batch_first
        self.end_on_each_frame = end_on_each_frame

        # valid check
        self._validate_parameters()

    def _validate_parameters(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.seq_length is not None:
            if self.seq_length <= 0:
                raise ValueError("seq_length must be positive")
            if any(frame_count < self.seq_length for frame_count in self.video_dataset.frame_counts):
                raise ValueError("All videos must have at least seq_length frames")
        else:
            if self.batch_size > 1:
                raise ValueError("if batch_size_per_gpu > 1, then seq_length should be provided in the dataset config.")
            if self.shuffle:
                raise ValueError("if use_shuffle is True, then seq_length should be provided in the dataset config.")

        if self.dynamic_batch_size and not self.batch_first:
            raise ValueError(
                "batch_first should be set to True when dynamic_batch_size is True. "
                "Otherwise, the first dimension of the batch will be the clip dimension, which is not supported."
            )

    def _generate_video_orders(self):
        # deterministically shuffle video order based on epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            video_indices = torch.randperm(self.video_dataset.video_count, generator=generator).tolist()
        else:
            video_indices = list(range(self.video_dataset.video_count))

        # subsample
        video_indices = video_indices[self.rank::self.num_replicas]

        # generate valid starts for each video
        video_orders = []
        if self.shuffle:
            for vid_idx in video_indices:
                max_start = self.video_dataset.frame_counts[vid_idx] - self.seq_length
                starts = torch.randint(0, max_start + 1, (1,), generator=generator).tolist()
                lengths = [self.seq_length] * len(starts)
                video_orders.append((vid_idx, starts, lengths))
        else:
            for vid_idx in video_indices:
                starts = [0]
                lengths = [self.seq_length or self.video_dataset.frame_counts[vid_idx]]
                video_orders.append((vid_idx, starts, lengths))

        return video_orders

    def _batch_indices_generator(self, video_orders):
        batch = []
        for vid_idx, starts, lengths in video_orders:
            for start, length in zip(starts, lengths):
                # generate contiguous frame indices from the current start
                frame_indices = list(range(start, start + length))
                # (video_idx, frame_idx, is_end_signal)
                if self.has_end_signal:
                    batch.extend([(vid_idx, idx, i == length - 1) for i, idx in enumerate(frame_indices)])
                else:
                    batch.extend([(vid_idx, idx, False) for idx in frame_indices])

                # yield when batch is full
                if len(batch) >= self.batch_size * length:
                    # (seq_length, batch_size, ...) -> (batch_size, seq_length, ...)
                    if self.batch_first:
                        batch = [
                            batch[i + j]
                            for j in range(length)
                            for i in range(0, len(batch), length)
                        ]

                    # (N * seq_length, ...) -> k * (batch_size * seq_length, ...)
                    yield from self._split_batch(batch)
                    batch = []

        # process the last uncompleted batch
        if len(batch) > 0 and not self.drop_last:
            yield from self._split_batch(batch)

    def _split_batch(self, batch):
        """
        Split a batch into smaller chunks of size batch_size * seq_length.

        """
        ### batch_size = 4
        # [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
        # -> [1, 5, 9, 13]
        # -> [2, 6, 10, 14]
        # -> [3, 7, 11, 15]
        # -> [4, 8, 12, 16]
        if self.batch_first and self.dynamic_batch_size:
            for i in range(0, len(batch), self.batch_size):
                chunk = batch[i:i + self.batch_size]
                if len(chunk) == self.batch_size:
                    for index in chunk:
                        yield index
        ### batch_size = 2, seq_length = 4
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16]
        # -> [1, 2, 3, 4, 5, 6, 7, 8]
        # -> [9, 10, 11, 12, 13, 14, 15, 16]
        elif self.seq_length is not None:
            for i in range(0, len(batch), self.batch_size * self.seq_length):
                chunk = batch[i:i + self.batch_size * self.seq_length]
                if len(chunk) == self.batch_size * self.seq_length:
                    for index in chunk:
                        yield index
        elif self.batch_size == 1:
            for index in batch:
                yield index
        else:
            raise ValueError("Unsupported combination of batch_size, seq_length, and dynamic_batch_size")

    def __iter__(self):
        video_orders = self._generate_video_orders()
        self.epoch += 1

        return iter(self._batch_indices_generator(video_orders))

    def __len__(self):
        total_clips = self.video_dataset.video_count
        frame_counts: list = self.video_dataset.frame_counts

        if self.seq_length is not None:
            total_samples = total_clips * self.seq_length
        else:
            total_samples = sum(frame_count for frame_count in frame_counts)

        # calculate distributed size
        left_samples = total_samples % self.num_replicas
        total_samples = total_samples // self.num_replicas + (1 if self.rank < left_samples else 0)

        # calculate dropped size
        if self.drop_last:
            if self.seq_length is not None:
                dropped_size = (total_clips * self.seq_length) % (self.batch_size * self.seq_length)
            else:
                dropped_size = 0
            total_samples -= dropped_size
        return total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# 完全利用视频中的所有帧，数据集需要相应缩小 seq_length 倍
# 使用这种方式不会很大程度上改变视频片段数据集的样本分布（序列越长效果越不明显）
# 其改变视频片段的样本分布按照平移的方式进行变化
# 比如：数据集中有视频片段（每个字母代表视频帧数） [a, b, c, d, ..., h], seq_length=s
# 则样本数量变为 [a-s+1, b-s+1, c-s+1, d-s+1, ..., h-s+1]，即都向下平移了 s-1 个单位
# 要想解决这种不平衡，可以再均匀随机从原视频帧中抽取 s-1 个视频片段，对于视频片段的样本分布就变得平衡了
"""
e.g. (where [i,j] represents i,i+1,i+2,...,j-2,j-1,j)
1,2,3,4,5,6,7,8,9,10,11,12          -> [1,7];[2,8];[3,9];[4,10];[5,11];[6,12]
1,2,3,4,5,6,7,8,9,10,11,12,13       -> [1,7];[2,8];[3,9];[4,10];[5,11];[6,12];[7,13];[8,14]
1,2,3,4,5,6,7,8,9,10,11,12,13,14    -> [1,7];[2,8];[3,9];[4,10];[5,11];[6,12];[7,13];[8,14]
1,2,3,4,5,6,7,8,9,10                -> [1,7];[2,8];[3,9];[4,10]
1,2,3,4,5,6,7,8,9,10,11,12          -> [1,7];[2,8];[3,9];[4,10];[5,11];[6,12]
shuffle
"""
@SAMPLER_REGISTRY.register()
class CompletedVideoClipSampler(Sampler):
    def __init__(
            self,
            video_dataset,
            num_replicas=1,
            rank=0,
            seed=0,

            seq_length=None,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            has_end_signal=False,
            dynamic_batch_size=False,
            batch_first=False,
            end_on_each_frame=False,

            max_clip_number=None,

            ratio=1,
    ):
        """
        Distributed Sampler for video clip training.

        Args:
            video_dataset: a VideoDataset object with frame_counts and video_count attribute.
            seq_length (int): length of each clip.
            batch_size (int): batch size.
            shuffle (bool): whether to shuffle the video order.
            drop_last (bool): whether to drop the last incomplete batch.
            has_end_signal (bool): whether to include end signal in the batch.
            dynamic_batch_size (bool): whether to return a frame-level batch or clip-level batch.
            batch_first (bool): should the first dimension be batch or clip.

            num_replicas (Optional[int]): number of processes participating in distributed training. Defaults to None.
            rank (Optional[int]): rank of the current process within num_replicas. Defaults to None.
            seed (int): random seed.

            max_clip_number (int): maximum number of clips to sample from all videos.
            ratio (int): enlarging ratio. Default: 1.
        """
        super().__init__(video_dataset)

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.ratio = ratio

        self.video_dataset = video_dataset
        # seq_length is None also means not in the training phase
        self.seq_length = seq_length    # if seq_length is None, set it to frame_count of each video
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.has_end_signal = has_end_signal
        self.dynamic_batch_size = dynamic_batch_size
        self.batch_first = batch_first
        self.end_on_each_frame = end_on_each_frame

        self.max_clip_number = max_clip_number

        # valid check
        self._validate_parameters()

    def _validate_parameters(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.seq_length is not None:
            if self.seq_length <= 0:
                raise ValueError("seq_length must be positive")
            if any(frame_count < self.seq_length for frame_count in self.video_dataset.frame_counts):
                raise ValueError("All videos must have at least seq_length frames")
        else:
            if self.batch_size > 1:
                raise ValueError("if batch_size_per_gpu > 1, then seq_length should be provided in the dataset config.")
            if self.shuffle:
                raise ValueError("if use_shuffle is True, then seq_length should be provided in the dataset config.")

        if self.dynamic_batch_size and not self.batch_first:
            raise ValueError(
                "batch_first should be set to True when dynamic_batch_size is True. "
                "Otherwise, the first dimension of the batch will be the clip dimension, which is not supported."
            )

    def _generate_video_orders(self):
        # deterministically shuffle video order based on epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        video_indices = list(range(self.video_dataset.video_count))

        # subsample
        video_indices = video_indices[self.rank::self.num_replicas]

        # generate valid starts for each video
        video_orders = []
        if self.shuffle:
            for vid_idx in video_indices:
                #: frame_count=10,seq_length=5 -> [0,1,2,3,4,5,0,3,7,5], length=5
                frame_count = self.video_dataset.frame_counts[vid_idx]
                valid_length = frame_count - self.seq_length + 1
                append_length = ((self.seq_length - 1) // valid_length + 1) * valid_length

                starts = list(range(valid_length))
                # valid_length > seq_length 时，溢出帧不重复（尽可能采样均衡）
                append_starts = torch.randperm(append_length, generator=generator).tolist()[:self.seq_length-1]
                append_starts = [start % valid_length for start in append_starts]
                starts.extend(append_starts)

                lengths = [self.seq_length] * len(starts)
                video_orders.extend([
                    (vid_idx, start, length) for start, length in zip(starts, lengths)
                ])
        else:
            for vid_idx in video_indices:
                #: frame_count=10,seq_length=5 -> [0,1,2,3,4,5,0,1,2,3], length=5
                if self.seq_length is not None:
                    frame_count = self.video_dataset.frame_counts[vid_idx]
                    # 数量还是 frame_count，但是限制最大值为 frame_count - self.seq_length + 1，超出的部分取模
                    starts = [i % (frame_count - self.seq_length + 1) for i in range(frame_count)]
                    lengths = [self.seq_length] * len(starts)
                #: frame_count=10,seq_length=5 -> [0], length=10
                else:
                    starts = [0]
                    lengths = [self.video_dataset.frame_counts[vid_idx]]
                video_orders.extend([
                    (vid_idx, start, length) for start, length in zip(starts, lengths)
                ])

        if self.shuffle:
            indices = torch.randperm(len(video_orders), generator=generator).tolist()
            video_orders = [video_orders[i] for i in indices]

        if self.max_clip_number is not None:
            video_orders = video_orders[:self.max_clip_number]

        # Apply ratio by repeating the video orders
        if self.ratio > 1:
            original_length = len(video_orders)
            video_orders = video_orders * self.ratio
            # Shuffle the repeated orders to mix them up
            if self.shuffle:
                indices = torch.randperm(len(video_orders), generator=generator).tolist()
                video_orders = [video_orders[i] for i in indices]
            else:
                # For non-shuffle case, just repeat the sequence
                pass

        return video_orders

    def _batch_indices_generator(self, video_orders):
        batch = []
        for vid_idx, start, length in video_orders:
            # generate contiguous frame indices from the current start
            frame_indices = list(range(start, start + length))
            # (video_idx, frame_idx, is_end_signal)
            if self.has_end_signal:
                batch.extend([(vid_idx, idx, i == length - 1) for i, idx in enumerate(frame_indices)])
            else:
                batch.extend([(vid_idx, idx, False) for idx in frame_indices])

            # yield when batch is full
            if len(batch) >= self.batch_size * length:
                # (seq_length, batch_size, ...) -> (batch_size, seq_length, ...)
                if self.batch_first:
                    batch = [
                        batch[i + j]
                        for j in range(length)
                        for i in range(0, len(batch), length)
                    ]

                # (N * seq_length, ...) -> k * (batch_size * seq_length, ...)
                yield from self._split_batch(batch)
                batch = []

        # process the last uncompleted batch
        if len(batch) > 0 and not self.drop_last:
            yield from self._split_batch(batch)

    def _split_batch(self, batch):
        """
        Split a batch into smaller chunks of size batch_size * seq_length.

        """
        ### batch_size = 4
        # [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
        # -> [1, 5, 9, 13]
        # -> [2, 6, 10, 14]
        # -> [3, 7, 11, 15]
        # -> [4, 8, 12, 16]
        if self.batch_first and self.dynamic_batch_size:
            for i in range(0, len(batch), self.batch_size):
                chunk = batch[i:i + self.batch_size]
                if len(chunk) == self.batch_size:
                    for index in chunk:
                        yield index
        ### batch_size = 2, seq_length = 4
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16]
        # -> [1, 2, 3, 4, 5, 6, 7, 8]
        # -> [9, 10, 11, 12, 13, 14, 15, 16]
        elif self.seq_length is not None:
            for i in range(0, len(batch), self.batch_size * self.seq_length):
                chunk = batch[i:i + self.batch_size * self.seq_length]
                if len(chunk) == self.batch_size * self.seq_length:
                    for index in chunk:
                        yield index
        elif self.batch_size == 1:
            for index in batch:
                yield index
        else:
            raise ValueError("Unsupported combination of batch_size, seq_length, and dynamic_batch_size")

    def __iter__(self):
        video_orders = self._generate_video_orders()
        self.epoch += 1

        return iter(self._batch_indices_generator(video_orders))

    def __len__(self):
        frame_counts: list = self.video_dataset.frame_counts

        if self.seq_length is not None:
            total_samples = sum([
                frame_count * self.seq_length for frame_count in frame_counts if frame_count >= self.seq_length
            ])
            # 如果不使用额外随机采样的方式，则数量为：
            # sum([
            #   (frame_count - self.seq_length + 1) * self.seq_length for frame_count in frame_counts if frame_count >= self.seq_length
            # ])
        else:
            total_samples = sum(frame_count for frame_count in frame_counts)

        # Apply ratio
        total_samples = total_samples * self.ratio

        # calculate distributed size
        total_samples = total_samples // self.num_replicas
        return total_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


"""
e.g. (where [i,j] represents i,i+1,i+2,...,j-2,j-1,j)
the same as CompletedVideoClipSampler, but sequence length varies across epochs
shuffle
"""
@SAMPLER_REGISTRY.register()
class ScheduledVideoClipSampler(CompletedVideoClipSampler):
    def __init__(
            self,
            *args,
            seq_length_schedule: dict,
            **kwargs
    ):
        self.seq_length_scheduler = get_schedule(seq_length_schedule)
        kwargs.update(dict(seq_length=self.seq_length_scheduler(1)))
        super().__init__(*args, **kwargs)

    def __iter__(self):
        frame_counts: list = self.video_dataset.frame_counts
        total_samples = sum(frame_count for frame_count in frame_counts)
        self.seq_length = self.seq_length_scheduler(self.epoch)
        self.max_clip_number = (total_samples - 1) // self.seq_length + 1
        return super().__iter__()

    def __len__(self):
        frame_counts: list = self.video_dataset.frame_counts
        total_samples = sum(frame_count for frame_count in frame_counts)

        total_samples = ((total_samples - 1) // self.seq_length + 1) * self.seq_length

        # calculate distributed size
        total_samples = total_samples // self.num_replicas
        return total_samples

    @staticmethod
    def monotonic_find_epoch_range(function, target, epoch_range, decreasing=False):
        """
        Find the range of epochs that the function takes the target value.
        Args:
            function: a function that takes epoch as input and returns a value, and must be monotonic.
            target: the target value.
            epoch_range: a tuple of (left, right) epoch indices.
            decreasing: whether the function is decreasing or increasing.
        Returns:
            a tuple of (left_bound, right_bound) epoch indices.
        """
        # find left bound
        left, right = epoch_range
        left_bound = right + 1  # 初始化为无效值
        while left <= right:
            mid = (left + right) // 2
            if decreasing ^ (function(mid) >= target):
                left_bound = mid
                right = mid - 1
            else:
                left = mid + 1

        # find right bound
        lef, right = epoch_range
        right_bound = left - 1  # 初始化为无效值
        while left <= right:
            mid = (left + right) // 2
            if decreasing ^ (function(mid) <= target):
                right_bound = mid
                left = mid + 1
            else:
                right = mid - 1

        return left_bound, right_bound

    @staticmethod
    def blind_find_epoch_range(function, epoch):
        """
        Find the range of epochs that the function takes the target value.
        Only look forwards from the current epoch.
        Args:
            function: a function that takes epoch as input and returns a value. Can be non-monotonic.
            epoch: the current epoch.
        Returns:
            a tuple of (left_bound, right_bound) epoch indices.
        """
        left = epoch
        target = function(left)
        while function(left) != target:
            left += 1
        right = left
        while function(left) == target:
            left -= 1
        while function(right) == target:
            right += 1
        return left + 1, right - 1


def get_schedule(seq_length_schedule: dict):
    """
    e.g.
    seq_length_schedule = {
        "type": "multistep",
        "epoch_steps": [5, 15, 30],         # or (5, 15, 30)
        "seq_length_steps": [10, 20, 30]    # or (10, 20, 30)
    }

    seq_length_schedule = {
        "type": "custom_cosine_schedule",
        "max_epoch": 500,
        "seq_length": 30,
    }
    or
    seq_length_schedule = {
        "type": "custom_cosine_schedule",
        "params": {
            "max_epoch": 500,
            "seq_length": 30,
        }
    }

    seq_length_schedule = {
        "type": "with_warmup_schedule",
        "max_epoch": 500,
        "seq_length": 30,
        "warmup_epoch": 10,
        "warmup_seq_length": 10,
        "base_function": {
            "type": "cosine_schedule",
        }
    }
    """
    schedule_function, schedule_params = parse_params(seq_length_schedule, 'constant_schedule')
    if isinstance(schedule_function, str):
        if schedule_function not in globals():
            schedule_function = f"{schedule_function}_schedule"
        if schedule_function not in globals():
            raise ValueError(f"Unsupported seq_length_schedule: {seq_length_schedule}")
        seq_length_scheduler = globals()[schedule_function](**schedule_params)
    elif isinstance(schedule_function, int):
        seq_length_scheduler = constant_schedule(seq_length=schedule_function)
    else:
        raise ValueError(f"Unsupported seq_length_schedule: {seq_length_schedule}")
    return seq_length_scheduler


def _check_and_get_schedule_function(function, **default_params):
    if isinstance(function, dict):
        function.update(dict(params=default_params))
        function = get_schedule(function)
    elif not callable(function):
        raise ValueError(f"Unsupported base_function: {function}")
    return function


# 用于装饰：
# 1. 输入范围为 [0, 1] 的函数，使其输入范围为 [0, max_epoch]
# 2. 输出范围为 [0, 1] 的函数，使其输出范围为 [1, seq_length]
def _normalize_to_epoch(max_epoch, seq_length):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(x):
            x = x / max_epoch
            y = func(x)
            if x <= 0:
                return 1
            elif x >= 1:
                return seq_length
            return max(1, math.ceil(seq_length * y))
        return wrapper
    return decorator


# longer min & max seq_length term
def cosine_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return (1 + math.cos(x * math.pi)) / 2
    return function


def reversed_cosine_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return (1 - math.cos(x * math.pi)) / 2
    return function


def custom_cosine_schedule(epoch_range, seq_length_range):
    l, r = epoch_range
    d, u = seq_length_range
    @_normalize_to_epoch(r - l, u - d)
    def function(x):
        return (1 - math.cos((x - l) * math.pi)) / 2 + d
    return function


# uniform each seq_length term
def linear_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return x
    return function


def reversed_linear_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return 1 - x
    return function


# longer max seq_length term
def cubic_bezier_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return (4 * x ** 3 - 3 * x ** 2 + 3 * x) / (4 - 9 * x + 9 * x ** 2)
    return function


def reversed_cubic_bezier_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        x = 1 - x
        return (4 * x ** 3 - 3 * x ** 2 + 3 * x) / (4 - 9 * x + 9 * x ** 2)
    return function


def uniform_cubic_bezier_schedule(max_epoch, seq_length):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return 3 * x ** 2 - 2 * x ** 3
    return function


# constant seq_length
def constant_schedule(max_epoch=1, seq_length=30):
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        return seq_length
    return function


# multi-step seq_length
def multistep_schedule(epoch_steps, seq_length_steps):
    def function(epoch):
        for i, step in reversed(list(enumerate(epoch_steps))):
            if epoch < step:
                return seq_length_steps[i]
        return seq_length_steps[-1]
    return function


# loop alternating seq_length
def alternating_schedule(seq_lengths):
    def function(epoch):
        return seq_lengths[epoch % len(seq_lengths)]
    return function


def with_warmup_schedule(max_epoch, seq_length, warmup_epoch, warmup_seq_length, base_function):
    base_function = _check_and_get_schedule_function(base_function, max_epoch=max_epoch, seq_length=seq_length)
    denormalized_base_function = base_function.__wrapped__
    l = warmup_epoch / max_epoch
    t = warmup_seq_length / seq_length
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        if x < l:
            return x / l * t
        else:
            return denormalized_base_function((x - l) / (1 - l)) * (1 - t) + t
    return function


def with_inverse_warmup_schedule(max_epoch, seq_length, warmup_epoch, base_function):
    base_function = _check_and_get_schedule_function(base_function, max_epoch=max_epoch, seq_length=seq_length)
    denormalized_base_function = base_function.__wrapped__
    @_normalize_to_epoch(max_epoch, seq_length)
    def function(x):
        r = (max_epoch - warmup_epoch) / max_epoch
        if x > r:
            return (1 - x) / (1 - r)
        else:
            return denormalized_base_function(x / r)
    return function



