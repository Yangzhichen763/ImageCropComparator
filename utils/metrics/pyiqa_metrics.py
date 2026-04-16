import os

try:
    import pyiqa
except ImportError:
    print("pyiqa is not installed. Please install it via 'pip install pyiqa' to use the IQA metrics.")

import math
import time
import threading
import queue
import concurrent.futures
from typing import Dict, List, Optional


import sys
sys.path.append(".")
from utils.registry import METRICS_REGISTRY

try:
    from utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

try:
    from .util import gt_mean
except:
    from utils.metrics.util import gt_mean

"""
Command usage: https://github.com/chaofengc/IQA-PyTorch?tab=readme-ov-file
pyiqa [metric_name(s)] -t [image_path or dir] -r [image_path or dir] --device [cuda or cpu] --verbose
"""


@METRICS_REGISTRY.register()
class IQA:
    def __init__(self, metric_type, any_gt_mean=False):
        super().__init__()
        self.metric_type = metric_type

        self.metric_func = pyiqa.create_metric(self.metric_type)
        self.any_gt_mean = any_gt_mean

    def __call__(self, *inputs):
        """
        Args:
            inputs (torch.Tensor): list of tensors of shape (N, C, H, W).
        """
        metric_func = self.metric_func
        metric_func.to(inputs[0].device)

        if metric_func.metric_mode == 'FR':
            assert len(inputs) == 2, f"Unsupported number of inputs for {self.metric_type} metric: {len(inputs)}"

            pred, gt = inputs
            if self.any_gt_mean:
                pred, gt = gt_mean(pred, gt)
            return metric_func(pred, gt)
        elif metric_func.metric_mode == 'NR':
            assert len(inputs) == 1, f"Unsupported number of inputs for {self.metric_type} metric: {len(inputs)}"

            return metric_func(*inputs)
        else:
            raise ValueError(f"Unsupported number of inputs: {len(inputs)}")


@METRICS_REGISTRY.register()
class IQAs:
    def __init__(self, *metric_types, **kwargs):
        super().__init__()
        self.metric_types = metric_types

        self.metric_funcs = [
            {
                'type': metric_type,
                'name': metric_type.split(' ')[0],
                'gt-mean': any('gt-mean' in token for token in metric_type.split()),
                'gt-mean:mode': next(
                    (v for k, v in {
                        'gt-mean-c': 'channel-wise',
                        'gt-mean-b': 'brightness',
                    }.items() if k in metric_type.split(' ')),
                    None
                )
            }
            for metric_type in self.metric_types
        ]
        self.traditional = kwargs.get('traditional', False)

        for d in self.metric_funcs:
            d['func'] = self._resolve_metric_ctor(d['name'])()

    def _resolve_metric_ctor(self, metric_name: str):
        if not self.traditional:
            return lambda: pyiqa.create_metric(metric_name)

        import utils.metrics as metrics
        traditional_ctor_dict = {
            'ssim': lambda: metrics.ssim.SSIM(),
            'psnr': lambda: metrics.psnr.PSNR(),
            'lpips': lambda: metrics.lpips.LPIPS(),
        }
        if metric_name not in traditional_ctor_dict:
            print(f"Warning: metric '{metric_name}' does not have a traditional implementation; using pyiqa version instead.")
            return lambda: pyiqa.create_metric(metric_name)
        return traditional_ctor_dict[metric_name]


    def __call__(self, *inputs):
        """
        Args:
            inputs (torch.Tensor): list of tensors of shape (N, C, H, W).
        """
        metrics = {}
        for d in self.metric_funcs:
            metric_type = d['type']
            metric_name = d['name']
            metric_func = d['func']

            try:
                metric_func.to(inputs[0].device)
            except:
                pass

            if metric_func.metric_mode == 'FR':
                assert len(inputs) >= 2, f"Unsupported number of inputs for {metric_name} metric: {len(inputs)}"

                pred, gt = inputs[:2]
                if d.get('gt-mean', False):
                    pred, gt = gt_mean(pred, gt, mode=d.get('gt-mean:mode', None))
                metrics[metric_type] = metric_func(pred, gt)
            elif metric_func.metric_mode == 'NR':
                assert len(inputs) >= 1, f"Unsupported number of inputs for {metric_name} metric: {len(inputs)}"

                pred, = inputs[:1]
                metrics[metric_type] = metric_func(pred)
            else:
                raise ValueError(f"Unsupported number of inputs: {len(inputs)}")

        return metrics


@METRICS_REGISTRY.register()
class MultiThreadIQAs:
    def __init__(self, *metric_types, **kwargs):
        """Multi-threaded IQA metric runner.

        Key design:
        - Each metric gets its own ThreadPoolExecutor with n threads (default 2).
        - For each metric, we keep an instance pool of metric objects of size n.
          This avoids sharing a single metric instance across threads (often not thread-safe).
        """
        super().__init__()
        self.metric_types = metric_types
        self.n_threads_per_metric = int(kwargs.get('n_threads_per_metric', 2))
        if self.n_threads_per_metric <= 0:
            raise ValueError(f"n_threads_per_metric must be positive, got {self.n_threads_per_metric}")

        self.device = kwargs.get('device', None)
        self.traditional = kwargs.get('traditional', False)

        self.metric_specs = [
            {
                'type': metric_type,
                'name': metric_type.split(' ')[0],
                'gt-mean': any('gt-mean' in token for token in metric_type.split()),
                'gt-mean:mode': next(
                    (v for k, v in {
                        'gt-mean-c': 'channel-wise',
                        'gt-mean-b': 'brightness',
                    }.items() if k in metric_type.split(' ')),
                    None
                )
            }
            for metric_type in self.metric_types
        ]

        # Per-metric executors and per-metric metric-instance pools.
        self._executors: Dict[str, concurrent.futures.ThreadPoolExecutor] = {}
        self._func_pools: Dict[str, queue.Queue] = {}
        self._func_mode: Dict[str, str] = {}

        for spec in self.metric_specs:
            metric_type = spec['type']
            metric_name = spec['name']

            ctor = self._resolve_metric_ctor(metric_name)
            func_pool: queue.Queue = queue.Queue(maxsize=self.n_threads_per_metric)
            metric_mode = None
            for _ in range(self.n_threads_per_metric):
                func = ctor()
                if metric_mode is None:
                    metric_mode = getattr(func, 'metric_mode', None)
                if self.device is not None:
                    try:
                        func.to(self.device)
                    except Exception:
                        pass
                func_pool.put(func)

            self._executors[metric_type] = concurrent.futures.ThreadPoolExecutor(max_workers=self.n_threads_per_metric)
            self._func_pools[metric_type] = func_pool
            self._func_mode[metric_type] = metric_mode or 'FR'

    def _resolve_metric_ctor(self, metric_name: str):
        if not self.traditional:
            return lambda: pyiqa.create_metric(metric_name)

        import utils.metrics as metrics
        traditional_ctor_dict = {
            'ssim': lambda: metrics.ssim.SSIM(),
            'psnr': lambda: metrics.psnr.PSNR(),
            'lpips': lambda: metrics.lpips.LPIPS(),
        }
        if metric_name not in traditional_ctor_dict:
            print(f"Warning: metric '{metric_name}' does not have a traditional implementation; using pyiqa version instead.")
            return lambda: pyiqa.create_metric(metric_name)
        return traditional_ctor_dict[metric_name]

    def shutdown(self, wait: bool = True):
        for ex in self._executors.values():
            ex.shutdown(wait=wait)

    def submit(self, pred, gt, gt_path: str, metrics_pool: dict, pool_lock: threading.Lock):
        """Submit all metrics for one sample.

        Updates metrics_pool[gt_path][metric_type] with float results (or NaN on failure).
        Returns list of futures.
        """
        futures: List[concurrent.futures.Future] = []
        for spec in self.metric_specs:
            metric_type = spec['type']
            executor = self._executors[metric_type]
            fut = executor.submit(
                self._compute_one_and_update,
                metric_type,
                spec,
                pred,
                gt,
                gt_path,
                metrics_pool,
                pool_lock,
            )
            futures.append(fut)
        return futures

    def __call__(self, *inputs):
        """Synchronous convenience wrapper.

        Keeps compatibility with the old call style: returns a dict mapping metric_type -> float.
        """
        assert len(inputs) >= 1
        pred = inputs[0]
        gt = inputs[1] if len(inputs) >= 2 else None

        if gt is None:
            any_fr = any(self._func_mode.get(mt, 'FR') == 'FR' for mt in self.metric_types)
            assert not any_fr, "FR metrics require (pred, gt) inputs"

        gt_path = '__single__'
        metrics_pool = {gt_path: {mt: None for mt in self.metric_types}}
        lock = threading.Lock()
        futures = self.submit(pred, gt, gt_path, metrics_pool, lock)
        if futures:
            concurrent.futures.wait(futures)
        return metrics_pool[gt_path]

    def _compute_one_and_update(self, metric_type, spec, pred, gt, gt_path, metrics_pool, pool_lock: threading.Lock):
        func_pool = self._func_pools[metric_type]
        metric_func = func_pool.get()
        try:
            if self.device is None:
                try:
                    metric_func.to(pred.device)
                except Exception:
                    pass

            metric_mode = getattr(metric_func, 'metric_mode', self._func_mode.get(metric_type, 'FR'))
            if metric_mode == 'FR':
                pred_i, gt_i = pred, gt
                if spec.get('gt-mean', False):
                    pred_i, gt_i = gt_mean(pred_i, gt_i, mode=spec.get('gt-mean:mode', None))
                out = metric_func(pred_i, gt_i)
            elif metric_mode == 'NR':
                out = metric_func(pred)
            else:
                raise ValueError(f"Unsupported metric_mode={metric_mode} for metric_type={metric_type}")

            if hasattr(out, 'item'):
                value = float(out.item())
            else:
                value = float(out)
        except Exception as e:
            value = float('nan')
            print(f"Error computing metric '{metric_type}' for gt_path '{gt_path}': {e}")
        finally:
            func_pool.put(metric_func)

        with pool_lock:
            # Dynamic update; caller may or may not have pre-initialized keys.
            if gt_path not in metrics_pool:
                metrics_pool[gt_path] = {}
            metrics_pool[gt_path][metric_type] = value

        return value


def metrics_to_str(name, metrics, max_name_len=None):
    try:
        import torch
    except Exception:
        torch = None

    metrics_strs = [
        f'{CP.keyword(k)}: {CP.number(f"{v.item():.4f}")}' if (torch is not None and isinstance(v, torch.Tensor)) else f'{CP.keyword(k)}: {CP.number(f"{v:.4f}")}'
        for k, v in metrics.items()
    ]
    metrics_trimmed_strs = [
        f'{k}: {v.item():.4f}' if (torch is not None and isinstance(v, torch.Tensor)) else f'{k}: {v:.4f}'
        for k, v in metrics.items()
    ]
    if max_name_len is None:
        out_str = f"{name}: {', '.join([f'{s:<{16 + len(s) - len(t_s)}}' for t_s, s in zip(metrics_trimmed_strs, metrics_strs)])}"
    else:
        out_str = f"{name:<{max_name_len}}: {', '.join([f'{s:<{16 + len(s) - len(t_s)}}' for t_s, s in zip(metrics_trimmed_strs, metrics_strs)])}"
    return out_str


def compute_iqa_metrics(
        dataroot_pred, dataroot_gt,
        metric_types=("psnr", "ssim", "lpips", "niqe", "brisque", "nima", "musiq", "pi", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean", "psnr gt-mean-c", "ssim gt-mean-c", "lpips gt-mean-c"),
        # "nima", "musiq", "pi", "maniqa", "clipiqa", "dists"
        verbose=True,
        **kwargs
):
    import torch
    import shutil
    from datetime import datetime
    from utils.datasets.simple_glob_dataset import PairedImageDataset, ImageDataset

    no_reference_mode = dataroot_gt is None
    if no_reference_mode:
        dataset = ImageDataset(dataroot_pred=dataroot_pred)
    else:
        dataset = PairedImageDataset(dataroot_pred=dataroot_pred, dataroot_gt=dataroot_gt)

    iqa = IQAs(*metric_types, **kwargs)
    fr_metric_types = [d['type'] for d in iqa.metric_funcs if getattr(d['func'], 'metric_mode', None) == 'FR']
    if no_reference_mode:
        nr_metric_types = [d['type'] for d in iqa.metric_funcs if getattr(d['func'], 'metric_mode', None) == 'NR']
        if fr_metric_types:
            print(f"Warning: no reference provided, FR metrics will be set to NaN: {', '.join(fr_metric_types)}")
        iqa = IQAs(*nr_metric_types, **kwargs) if nr_metric_types else None


    results = {
        key: [] for key in metric_types
    }
    max_name_len = max([len(os.path.basename(data['pred']['path'])) for data in dataset] + [len('average')])
    start_t = time.time()
    for data in dataset:
        pred = data['pred']['image'].unsqueeze(0).cuda()
        gt = None
        if not no_reference_mode:
            gt = data['gt']['image'].unsqueeze(0).cuda()
            if pred.shape != gt.shape:
                h, w = gt.shape[-2:]
                pred = pred[:, :, :h, :w]
                print(f"Warning: pred and gt shapes differ for {data['pred']['path']}; cropped pred to {pred.shape}")
        image_name = os.path.basename(data['pred']['path'])

        # Compute the IQA metrics
        if no_reference_mode:
            metrics = iqa(pred) if iqa is not None else {}
            for metric_type in fr_metric_types:
                metrics[metric_type] = float('nan')
        else:
            metrics = iqa(pred, gt)

        # Collect the results
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                results[k].append(float(v.item()))
            else:
                results[k].append(float(v))

        if verbose:
            # Print verbose output
            print(metrics_to_str(image_name, metrics, max_name_len=max_name_len))

    # Calculate the average values
    avg_results = {
        k: sum(v) / len(v) for k, v in results.items()
    }

    # Print the final results
    screen_width = shutil.get_terminal_size().columns
    if verbose:
        elapsed = time.time() - start_t
        ts = f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed: {elapsed:.2f}s "
        print(ts.center(screen_width, "="))
    print(metrics_to_str('average', avg_results))
    return avg_results


def compute_iqa_metrics_multi_thread(
        dataroot_pred, dataroot_gt,
        metric_types=
        ("psnr", "ssim", "lpips", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean"),
        # ("brisque", "niqe"),
        # ("psnr", "ssim", "lpips", "niqe", "brisque", "nima", "musiq", "pi", "psnr gt-mean", "ssim gt-mean", "lpips gt-mean", "psnr gt-mean-c", "ssim gt-mean-c", "lpips gt-mean-c"),
        # "nima", "musiq", "pi", "maniqa", "clipiqa", "dists"
        verbose=True,
        n_threads_per_metric: int = 2,
        poll_interval_s: float = 0.1,
        transform=None,
        reuse_iqa_model=False,
        iqa_model=None,
        **kwargs
):
    import torch
    import shutil
    from datetime import datetime
    from collections import deque
    from utils.datasets.simple_glob_dataset import PairedImageDataset, ImageDataset

    if poll_interval_s <= 0:
        raise ValueError(f"poll_interval_s must be > 0, got {poll_interval_s}")

    no_reference_mode = dataroot_gt is None
    if no_reference_mode:
        dataset = ImageDataset(dataroot_pred=dataroot_pred)
    else:
        dataset = PairedImageDataset(dataroot_pred=dataroot_pred, dataroot_gt=dataroot_gt)

    probe_iqa = IQAs(*metric_types, **kwargs)
    fr_metric_types = [d['type'] for d in probe_iqa.metric_funcs if getattr(d['func'], 'metric_mode', None) == 'FR']
    if no_reference_mode:
        metric_types_eval = tuple(
            d['type'] for d in probe_iqa.metric_funcs if getattr(d['func'], 'metric_mode', None) == 'NR'
        )
        if fr_metric_types:
            print(f"Warning: no reference provided, FR metrics will be set to NaN: {', '.join(fr_metric_types)}")
    else:
        metric_types_eval = metric_types

    kwargs_for_runner = dict(kwargs)
    device_from_kwargs = kwargs_for_runner.pop('device', None)
    if device_from_kwargs is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(device_from_kwargs)
    if iqa_model is not None:
        iqa = iqa_model
    else:
        iqa = MultiThreadIQAs(
            *metric_types_eval,
            device=device,
            n_threads_per_metric=n_threads_per_metric,
            **kwargs_for_runner,
        ) if len(metric_types_eval) > 0 else None

    # 指标池：key=gt图像路径，value={metric_type: value(None/float)}，动态更新
    metrics_pool: Dict[str, Dict[str, Optional[float]]] = {}
    pool_lock = threading.Lock()

    # queue：按读取顺序存储gt图像路径（支持peek）
    path_queue = deque()
    queue_lock = threading.Lock()
    reader_done = threading.Event()

    results: Dict[str, List[float]] = {k: [] for k in metric_types}
    max_name_len = len('average')

    all_futures: List[concurrent.futures.Future] = []

    def reader_worker():
        nonlocal max_name_len
        try:
            for data in dataset:
                pred = data['pred']['image'].unsqueeze(0).to(device)
                if no_reference_mode:
                    gt = None
                    gt_path = data['pred']['path']
                else:
                    gt = data['gt']['image'].unsqueeze(0).to(device)
                    if pred.shape != gt.shape:
                        h, w = gt.shape[-2:]
                        pred = pred[:, :, :h, :w]
                        print(f"Warning: pred and gt shapes differ for {data['pred']['path']}; cropped pred to {pred.shape}")
                    gt_path = data['gt']['path']

                # 动态创建 key；value 字典也按该样本动态建立/扩展
                with pool_lock:
                    if gt_path not in metrics_pool:
                        metrics_pool[gt_path] = {}
                    # 这里仅为该gt_path初始化需要的指标key，默认None（不是一次性初始化所有图片key）
                    for metric_type in metric_types:
                        if no_reference_mode and metric_type in fr_metric_types:
                            metrics_pool[gt_path].setdefault(metric_type, float('nan'))
                        else:
                            metrics_pool[gt_path].setdefault(metric_type, None)

                with queue_lock:
                    path_queue.append(gt_path)

                max_name_len = max(max_name_len, len(os.path.basename(gt_path)))
                if transform:
                    pred, gt = transform(pred, gt)
                if iqa is not None:
                    all_futures.extend(iqa.submit(pred, gt, gt_path, metrics_pool, pool_lock))
        finally:
            reader_done.set()

    start_t = time.time()
    t = threading.Thread(target=reader_worker, name='iqa_reader', daemon=True)
    t.start()

    # 每隔0.1秒检查一遍queue peek；如peek对应的指标都算完->出队->print->立刻检查下一个peek
    while True:
        time.sleep(poll_interval_s)

        while True:
            with queue_lock:
                head = path_queue[0] if len(path_queue) > 0 else None

            if head is None:
                break

            with pool_lock:
                head_metrics = metrics_pool.get(head, {})
                ready = all(head_metrics.get(mt, None) is not None for mt in metric_types)
                metrics_snapshot = dict(head_metrics) if ready else None

            if not ready:
                break

            with queue_lock:
                if len(path_queue) > 0 and path_queue[0] == head:
                    path_queue.popleft()

            # 收集结果并打印
            for k, v in metrics_snapshot.items():
                if v is None:
                    continue
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    continue
                results[k].append(float(v))

            if verbose:
                print(metrics_to_str(os.path.basename(head), metrics_snapshot, max_name_len=max_name_len))

        if reader_done.is_set():
            with queue_lock:
                queue_empty = len(path_queue) == 0
            if queue_empty:
                break

    # 确保任务结束并释放线程池
    if all_futures:
        concurrent.futures.wait(all_futures)
    if (not reuse_iqa_model) and (iqa is not None):
        iqa.shutdown(wait=True)

    # 计算平均（忽略 NaN/Inf）
    avg_results = {}
    for k, vals in results.items():
        clean = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
        avg_results[k] = (sum(clean) / len(clean)) if len(clean) > 0 else float('nan')

    # Print the final results
    screen_width = shutil.get_terminal_size().columns
    if verbose:
        elapsed = time.time() - start_t
        ts = f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | elapsed: {elapsed:.2f}s "
        print(ts.center(screen_width, "="))
    print(metrics_to_str('average', avg_results))
    print("quick copy:", ' '.join([f'{v:.4f}' for k, v in avg_results.items()]))

    if reuse_iqa_model:
        return avg_results, iqa
    return avg_results


if __name__ == '__main__':
    """
    usage: 
    python basic/metrics/pyiqa_metrics.py -i <predicted_images_folder> -r <ground_truth_images_folder>] -g <gpu_indices>] [--traditional] [--multi-thread] [--num-threads <n>]
    """
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description="Compute image quality assessment (IQA) metrics between predicted and ground truth images."
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        type=str,
        nargs='+',
        help="One or more predicted image folders (supports glob patterns)."
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        default=None,
        help="Path to the folder containing ground truth images."
    )
    # parser.add_argument(
    #     '--verbose',
    #     action='store_true',
    #     help="Enable verbose output."
    # )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Disable verbose output."
    )
    parser.add_argument(
        '--gpus', '-g',
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to use."
    )
    parser.add_argument(
        '--traditional', '-t',
        action='store_true',
        help="Use traditional implementation instead of the iqa one."
    )
    parser.add_argument(
        '--multi-thread', '-m',
        action='store_true',
        help="Use multi-threaded IQA computation."
    )
    parser.add_argument(
        '--num-threads', '-n',
        type=int,
        default=2,
        help="Number of threads per metric for multi-threaded computation."
    )

    args = parser.parse_args()

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    import torch

    # Expand input glob(s), de-duplicate while preserving order.
    expanded_inputs = []
    for item in args.input:
        matches = sorted(glob.glob(item))
        if matches:
            expanded_inputs.extend([m for m in matches if os.path.isdir(m)])
        elif os.path.isdir(item):
            expanded_inputs.append(item)
        else:
            print(f"Warning: input path/pattern not found or not a directory: {item}")

    dedup_inputs = []
    seen = set()
    for p in expanded_inputs:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            dedup_inputs.append(p)

    if len(dedup_inputs) == 0:
        raise ValueError("No valid input directories found from --input/-i.")

    single_input_mode = len(dedup_inputs) == 1
    verbose = (not args.quiet) if single_input_mode else False

    if args.reference is None:
        print("No reference folder provided. Running in no-reference mode (FR metrics will be NaN).")

    all_results = {}
    for input_dir in dedup_inputs:
        print(f"\n=== Evaluating: {input_dir} ===")
        if args.multi_thread:
            df = compute_iqa_metrics_multi_thread(
                input_dir,
                args.reference,
                verbose=verbose,
                traditional=args.traditional,
                n_threads_per_metric=args.num_threads,
            )
        else:
            df = compute_iqa_metrics(
                input_dir,
                args.reference,
                verbose=verbose,
                traditional=args.traditional,
            )
            print("quick copy:", ' '.join([f'{v:.4f}' for k, v in df.items()]))

        all_results[input_dir] = df

    if len(all_results) > 1:
        print("\n" + "=" * 30 + " Batch Summary " + "=" * 30)
        for input_dir, avg_results in all_results.items():
            print(f"{input_dir}", ' '.join([f'{v:.4f}' for k, v in avg_results.items()]))
