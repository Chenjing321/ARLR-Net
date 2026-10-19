import argparse
import csv
import os
import os.path as osp
import traceback
from collections.abc import Mapping

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings
from torch.profiler import ProfilerActivity, profile

import arlr.datasets  # noqa: F401
import arlr.evaluation  # noqa: F401
import arlr.models  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description='Profile params and FLOPs with the real test dataloader.')
    parser.add_argument('--config', help='Config file for a single model.')
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Optional checkpoint for the single model.')
    parser.add_argument(
        '--config-list',
        default=None,
        help='CSV-like list: method,config,checkpoint(optional).')
    parser.add_argument('--device', default='cuda:0', help='Device to profile.')
    parser.add_argument(
        '--num-batches',
        type=int,
        default=1,
        help='Number of real test batches to profile.')
    parser.add_argument(
        '--out',
        default='work_dir/complexity_results.csv',
        help='Output CSV path.')
    args = parser.parse_args()

    if args.config is None and args.config_list is None:
        parser.error('Please specify either --config or --config-list.')
    if args.config is not None and args.config_list is not None:
        parser.error('Please specify only one of --config and --config-list.')
    if args.num_batches <= 0:
        parser.error('--num-batches must be a positive integer.')
    return args


def read_config_list(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) < 2 or len(parts) > 3:
                raise ValueError(
                    f'Invalid config-list format at line {line_no}: {line}')
            method, config = parts[0], parts[1]
            checkpoint = parts[2] if len(parts) == 3 and parts[2] else None
            if not method or not config:
                raise ValueError(
                    f'Method and config are required at line {line_no}: {line}')
            entries.append((method, config, checkpoint))
    return entries


def describe_value(obj):
    desc = f'type={type(obj).__name__}'
    if torch.is_tensor(obj):
        desc += f', shape={tuple(obj.shape)}, dtype={obj.dtype}'
    elif isinstance(obj, (list, tuple)):
        desc += f', len={len(obj)}'
        if len(obj) > 0:
            first = obj[0]
            desc += f', first_type={type(first).__name__}'
            if torch.is_tensor(first):
                desc += f', first_shape={tuple(first.shape)}'
    elif isinstance(obj, Mapping):
        desc += f', keys={list(obj.keys())}'
    return desc


def print_data_batch_summary(data_batch):
    print('\nData batch summary:')
    print(f'  data_batch type: {type(data_batch).__name__}')
    if isinstance(data_batch, Mapping):
        keys = list(data_batch.keys())
        print(f'  keys: {keys}')
        if 'inputs' in data_batch:
            print(f'  inputs: {describe_value(data_batch["inputs"])}')
        else:
            print('  inputs: <missing>')
        if 'data_samples' in data_batch:
            print(f'  data_samples: {describe_value(data_batch["data_samples"])}')
        else:
            print('  data_samples: <missing>')

        for key in keys[:8]:
            print(f'  field[{key}]: {describe_value(data_batch[key])}')
        if len(keys) > 8:
            print(f'  ... {len(keys) - 8} more fields omitted')
    else:
        print(f'  value: {describe_value(data_batch)}')


def looks_like_sample_sequence(obj):
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        return False
    if all(isinstance(item, (int, float, bool)) for item in obj):
        return False
    return True


def infer_batch_size(data_batch):
    """Infer batch size without mistaking a single sample tensor dim for B."""
    if isinstance(data_batch, Mapping):
        data_samples = data_batch.get('data_samples', None)
        if isinstance(data_samples, (list, tuple)):
            return len(data_samples)

        inputs = data_batch.get('inputs', None)
        if isinstance(inputs, (list, tuple)):
            return len(inputs)
        if torch.is_tensor(inputs):
            return int(inputs.shape[0])

        # Prefer sample-level sequences before tensor shapes when recursing.
        for value in data_batch.values():
            if looks_like_sample_sequence(value):
                return len(value)
        for value in data_batch.values():
            if torch.is_tensor(value) and value.dim() > 0:
                return int(value.shape[0])
        for value in data_batch.values():
            batch_size = infer_batch_size(value)
            if batch_size is not None:
                return batch_size
        return None

    if isinstance(data_batch, (list, tuple)):
        if len(data_batch) == 0:
            return 0
        return len(data_batch)

    if torch.is_tensor(data_batch) and data_batch.dim() > 0:
        return int(data_batch.shape[0])

    return None


def count_params(model):
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad)
    return total_params, trainable_params


def sync_arlr_runtime_device(model, device):
    """ARLR stores some tensors outside buffers, so move them explicitly."""
    torch_device = torch.device(device)
    model.to(torch_device)

    try:
        import arlr.models.editor.arlr as arlr_editor
        arlr_editor.device = torch_device
    except Exception:
        pass

    for name in ('Phi', 'Qinit'):
        value = getattr(model, name, None)
        if torch.is_tensor(value):
            setattr(model, name, value.to(torch_device))


def cuda_synchronize_if_needed(device):
    torch_device = torch.device(device)
    if torch_device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def build_runner(config):
    cfg = Config.fromfile(config)
    if cfg.get('custom_imports', None):
        import_modules_from_strings(**cfg.custom_imports)

    cfg.launcher = 'none'
    cfg.load_from = None
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(config))[0])

    return Runner.from_cfg(cfg)


def profile_one_batch(model, data_batch, device):
    activities = [ProfilerActivity.CPU]
    torch_device = torch.device(device)
    if torch_device.type == 'cuda' and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    model.eval()

    # Warmup is intentionally outside profiler.
    with torch.no_grad():
        _ = model.test_step(data_batch)
    cuda_synchronize_if_needed(device)

    with profile(
            activities=activities,
            record_shapes=True,
            with_flops=True,
    ) as prof:
        with torch.no_grad():
            _ = model.test_step(data_batch)
    cuda_synchronize_if_needed(device)

    total_flops = 0
    for item in prof.key_averages():
        flops = getattr(item, 'flops', None)
        if flops is not None:
            total_flops += flops
    return total_flops


def run_complexity(method, config, checkpoint, args):
    print('\n' + '=' * 80)
    print(f'Method: {method}')
    print(f'Config: {config}')
    print(f'Checkpoint: {checkpoint or ""}')

    runner = build_runner(config)
    model = runner.model

    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')

    sync_arlr_runtime_device(model, args.device)

    total_params, trainable_params = count_params(model)

    dataloader_iter = iter(runner.test_dataloader)
    profiled_batches = 0
    first_batch_size = None
    batch_sizes = []
    total_profiled_flops = 0
    total_profiled_samples = 0

    for batch_idx in range(args.num_batches):
        try:
            data_batch = next(dataloader_iter)
        except StopIteration as exc:
            raise RuntimeError(
                f'test_dataloader ended after {profiled_batches} batches') from exc

        if batch_idx == 0:
            print_data_batch_summary(data_batch)

        batch_size = infer_batch_size(data_batch)
        if batch_size is None or batch_size <= 0:
            raise RuntimeError(
                'Failed to infer a positive batch size from data_batch.')

        if first_batch_size is None:
            first_batch_size = batch_size
        batch_sizes.append(batch_size)

        total_flops = profile_one_batch(model, data_batch, args.device)
        total_profiled_flops += total_flops
        total_profiled_samples += batch_size
        profiled_batches += 1

    avg_flops_per_batch = total_profiled_flops / profiled_batches
    flops_per_sample = total_profiled_flops / total_profiled_samples

    if len(set(batch_sizes)) > 1:
        print(f'Warning: variable batch sizes were observed: {batch_sizes}')

    result = {
        'method': method,
        'config': config,
        'checkpoint': checkpoint or '',
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_M': total_params / 1e6,
        'batch_size': first_batch_size,
        'total_flops_G': avg_flops_per_batch / 1e9,
        'flops_per_sample_G': flops_per_sample / 1e9,
        'status': 'OK',
        'error_message': '',
    }

    print('\nResult:')
    print(f'Config: {config}')
    print(f'Checkpoint: {checkpoint or ""}')
    print(f'Total Params: {total_params}')
    print(f'Trainable Params: {trainable_params}')
    print(f'Params: {total_params / 1e6:.3f}M')
    print(f'\nBatch size: {first_batch_size}')
    print(
        f'Total FLOPs for one batch: {avg_flops_per_batch / 1e9:.4f} G')
    print(f'FLOPs per sample: {flops_per_sample / 1e9:.4f} G')
    print('\nImportant:')
    print('Use FLOPs per sample for paper table.')
    print(
        'FLOPs may be underestimated because unsupported ops are ignored by '
        'torch.profiler.')

    return result


def error_result(method, config, checkpoint, exc):
    error_message = ''.join(
        traceback.format_exception_only(type(exc), exc)).strip()
    return {
        'method': method,
        'config': config,
        'checkpoint': checkpoint or '',
        'total_params': '',
        'trainable_params': '',
        'params_M': '',
        'batch_size': '',
        'total_flops_G': '',
        'flops_per_sample_G': '',
        'status': 'ERROR',
        'error_message': error_message,
    }


def write_csv(path, rows):
    fieldnames = [
        'method',
        'config',
        'checkpoint',
        'total_params',
        'trainable_params',
        'params_M',
        'batch_size',
        'total_flops_G',
        'flops_per_sample_G',
        'status',
        'error_message',
    ]
    out_dir = osp.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if args.config_list:
        entries = read_config_list(args.config_list)
    else:
        method = osp.splitext(osp.basename(args.config))[0]
        entries = [(method, args.config, args.checkpoint)]

    rows = []
    has_error = False
    for method, config, checkpoint in entries:
        try:
            rows.append(run_complexity(method, config, checkpoint, args))
        except Exception as exc:
            has_error = True
            print('\nERROR:')
            print(''.join(traceback.format_exception(type(exc), exc,
                                                     exc.__traceback__)))
            rows.append(error_result(method, config, checkpoint, exc))
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.out, rows)
    print(f'\nSaved CSV to: {args.out}')

    if has_error and not args.config_list:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
