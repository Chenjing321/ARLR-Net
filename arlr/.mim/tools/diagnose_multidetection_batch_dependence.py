import argparse
import csv
import importlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from arlr.datasets import *  # noqa: F401,F403
from arlr.evaluation import *  # noqa: F401,F403
from arlr.models import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Diagnose duplicate detections and DynamicKernelMixture batch '
            'dependence.'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='Maximum test samples for detection statistics; 0 means all.')
    parser.add_argument('--num-probes', type=int, default=5)
    parser.add_argument('--num-contexts', type=int, default=3)
    parser.add_argument('--match-distance', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def get_first_evaluator_cfg(cfg):
    evaluator = cfg.get('test_evaluator', cfg.get('val_evaluator', {}))
    if isinstance(evaluator, (list, tuple)):
        return evaluator[0] if evaluator else {}
    return evaluator or {}


def resolve_options(cfg):
    metric_cfg = get_first_evaluator_cfg(cfg)
    c = int(metric_cfg.get('c', cfg.model.get('c', 3)))
    score_thr = float(metric_cfg.get('brightness_threshold', 50.0))
    batch_size = int(cfg.test_dataloader.get('batch_size', 1))
    return c, score_thr, batch_size


def select_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_model_module_device(device):
    module_names = [
        'distacp.models.editor.distacp',
        'distacp.models.blocks.DISTCP_block',
    ]
    for module_name in module_names:
        module = importlib.import_module(module_name)
        if hasattr(module, 'device'):
            module.device = device


def unwrap_output(outputs):
    if isinstance(outputs, (list, tuple)):
        if not outputs:
            raise RuntimeError('Model returned an empty prediction list.')
        outputs = outputs[0]
    if not isinstance(outputs, dict) or 'x_final' not in outputs:
        raise RuntimeError('Predict output must contain output["x_final"].')
    return outputs


def get_batch_item(value, index):
    if isinstance(value, (torch.Tensor, np.ndarray, list, tuple)):
        return value[index]
    return value


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_gt_array(targets):
    targets = np.asarray(targets, dtype=np.float64)
    if targets.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return targets.reshape(-1, 3)


def candidate_array(x_map, threshold, c, local_max=False):
    mask = x_map >= threshold
    if local_max:
        padded = np.pad(
            x_map, ((1, 1), (1, 1)),
            mode='constant',
            constant_values=-np.inf)
        neighborhood_max = np.full_like(x_map, -np.inf)
        for row_shift in range(3):
            for col_shift in range(3):
                neighborhood_max = np.maximum(
                    neighborhood_max,
                    padded[row_shift:row_shift + x_map.shape[0],
                           col_shift:col_shift + x_map.shape[1]])
        mask &= x_map >= neighborhood_max

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    center = (c - 1) // 2
    return np.stack([
        (rows.astype(np.float64) - center) / c,
        (cols.astype(np.float64) - center) / c,
        x_map[rows, cols].astype(np.float64),
    ], axis=1)


def nearest_gt_distances(candidates, gt):
    if candidates.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if gt.shape[0] == 0:
        return np.full((candidates.shape[0],), np.inf, dtype=np.float64)
    differences = candidates[:, None, :2] - gt[None, :, :2]
    return np.linalg.norm(differences, axis=2).min(axis=1)


def evaluation_style_matching(candidates, gt, match_distance):
    """Match like CSO_tpfp: score order, nearest GT, one use per GT."""
    nearest_distances = nearest_gt_distances(candidates, gt)
    matched = np.zeros((candidates.shape[0],), dtype=bool)
    if candidates.shape[0] == 0 or gt.shape[0] == 0:
        return matched, nearest_distances

    differences = candidates[:, None, :2] - gt[None, :, :2]
    distances = np.linalg.norm(differences, axis=2)
    nearest_gt = distances.argmin(axis=1)
    gt_covered = np.zeros((gt.shape[0],), dtype=bool)
    for candidate_index in np.argsort(-candidates[:, 2]):
        gt_index = nearest_gt[candidate_index]
        if (nearest_distances[candidate_index] <= match_distance
                and not gt_covered[gt_index]):
            matched[candidate_index] = True
            gt_covered[gt_index] = True
    return matched, nearest_distances


def format_distances(values):
    return ';'.join(
        'inf' if not np.isfinite(value) else f'{value:.6f}'
        for value in values)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return math.inf if numerator > 0 else 0.0
    return numerator / denominator


def analyze_sample(output, batch_index, sample_index, c, score_thr,
                   match_distance):
    x_final = to_numpy(output['x_final'][batch_index]).reshape(
        11 * c, 11 * c)
    gt = as_gt_array(output['targets_GT'][batch_index])
    raw_candidates = candidate_array(x_final, score_thr, c)
    local_max_candidates = candidate_array(
        x_final, score_thr, c, local_max=True)
    matched, nearest_distances = evaluation_style_matching(
        raw_candidates, gt, match_distance)
    unmatched_distances = nearest_distances[~matched]
    file_id = str(get_batch_item(output['file_id'], batch_index))

    return {
        'sample_index': sample_index,
        'file_id': file_id,
        'gt_count': int(gt.shape[0]),
        'raw_candidate_count': int(raw_candidates.shape[0]),
        'local_max_candidate_count': int(local_max_candidates.shape[0]),
        'local_max_removed_count': int(
            raw_candidates.shape[0] - local_max_candidates.shape[0]),
        'candidate_gt_ratio': safe_ratio(
            raw_candidates.shape[0], gt.shape[0]),
        'is_over_detection': int(raw_candidates.shape[0] > gt.shape[0]),
        'matched_candidate_count': int(matched.sum()),
        'unmatched_candidate_count': int((~matched).sum()),
        'candidate_nearest_gt_distances': format_distances(
            nearest_distances),
        'unmatched_nearest_gt_distances': format_distances(
            unmatched_distances),
        '_nearest_distances': nearest_distances,
        '_unmatched_distances': unmatched_distances,
    }


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f'No rows available for {path.name}.')
    fieldnames = [key for key in rows[0] if not key.startswith('_')]
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row[key] for key in fieldnames
            })


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def distance_bin_counts(distances):
    distances = np.asarray(distances, dtype=np.float64)
    return {
        'lt_0.1': int(np.sum(distances < 0.1)),
        '0.1_to_0.2': int(np.sum(
            (distances >= 0.1) & (distances < 0.2))),
        '0.2_to_0.3': int(np.sum(
            (distances >= 0.2) & (distances < 0.3))),
        'ge_0.3': int(np.sum(distances >= 0.3)),
    }


def write_overall_stats(path, rows, score_thr, match_distance):
    gt_counts = np.asarray([row['gt_count'] for row in rows])
    raw_counts = np.asarray([row['raw_candidate_count'] for row in rows])
    local_counts = np.asarray([
        row['local_max_candidate_count'] for row in rows
    ])
    ratios = np.asarray([
        row['candidate_gt_ratio'] for row in rows
    ], dtype=np.float64)
    finite_ratios = ratios[np.isfinite(ratios)]
    all_distances = np.concatenate([
        row['_nearest_distances'] for row in rows
    ])
    unmatched_distances = np.concatenate([
        row['_unmatched_distances'] for row in rows
    ])
    bins = distance_bin_counts(unmatched_distances)
    unmatched_total = unmatched_distances.size

    lines = [
        'Multi-detection source diagnostics',
        f'samples={len(rows)}',
        f'brightness_threshold={score_thr:g}',
        f'one_to_one_match_distance={match_distance:g}',
        'matching_rule=score-descending nearest-GT matching, consistent '
        'with CSO_tpfp',
        f'mean_gt_count={gt_counts.mean():.6f}',
        f'mean_raw_candidate_count={raw_counts.mean():.6f}',
        f'mean_candidate_gt_ratio={finite_mean(finite_ratios):.6f}',
        f'over_detection_sample_ratio={np.mean(raw_counts > gt_counts):.6f}',
        f'total_raw_candidates={raw_counts.sum()}',
        f'total_unmatched_candidates={unmatched_total}',
        f'mean_candidate_nearest_gt_distance={finite_mean(all_distances):.6f}',
        '',
        'Unmatched candidate cumulative distance ratios',
        'distance_lt_0.1_ratio='
        f'{np.mean(unmatched_distances < 0.1) if unmatched_total else 0.0:.6f}',
        'distance_lt_0.2_ratio='
        f'{np.mean(unmatched_distances < 0.2) if unmatched_total else 0.0:.6f}',
        'distance_lt_0.3_ratio='
        f'{np.mean(unmatched_distances < 0.3) if unmatched_total else 0.0:.6f}',
        'distance_ge_0.3_ratio='
        f'{np.mean(unmatched_distances >= 0.3) if unmatched_total else 0.0:.6f}',
        '',
        'Unmatched candidate distance bins (mutually exclusive)',
    ]
    for name, count in bins.items():
        ratio = count / unmatched_total if unmatched_total else 0.0
        lines.append(f'{name}: count={count}, ratio={ratio:.6f}')

    removed = raw_counts - local_counts
    total_raw = int(raw_counts.sum())
    lines.extend([
        '',
        'Offline 3x3 local-maximum diagnostic only',
        f'mean_local_max_candidate_count={local_counts.mean():.6f}',
        f'mean_removed_candidate_count={removed.mean():.6f}',
        f'total_local_max_candidates={local_counts.sum()}',
        f'total_removed_candidates={removed.sum()}',
        'local_max_candidate_reduction_ratio='
        f'{removed.sum() / total_raw if total_raw else 0.0:.6f}',
        'note=local maximum filtering is not used by the formal evaluation',
    ])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build_batch(samples):
    required_keys = ['gt_img_11', 'gt', 'ann_path', 'file_id']
    missing = [
        key for key in required_keys
        if any(key not in sample for sample in samples)
    ]
    if missing:
        raise KeyError(f'Dataset samples are missing keys: {missing}')
    return {
        key: [sample[key] for sample in samples]
        for key in required_keys
    }


def top_candidate(x_map, threshold):
    candidate_mask = x_map >= threshold
    if not np.any(candidate_mask):
        return -1, -1, math.nan
    candidate_values = np.where(candidate_mask, x_map, -np.inf)
    flat_index = int(np.argmax(candidate_values))
    row, col = np.unravel_index(flat_index, x_map.shape)
    return int(row), int(col), float(x_map[row, col])


def context_indices(dataset_size, probe_index, batch_size, rng):
    available = [index for index in range(dataset_size)
                 if index != probe_index]
    if len(available) < batch_size - 1:
        raise RuntimeError(
            f'Batch dependence test requires at least {batch_size} samples.')
    return [probe_index] + rng.sample(available, batch_size - 1)


def infer_probe_in_context(model, dataset, indices, c, score_thr):
    samples = [dataset[index] for index in indices]
    output = unwrap_output(model(mode='predict', **build_batch(samples)))
    x_map = to_numpy(output['x_final'][0]).reshape(11 * c, 11 * c)
    candidate_count = int(np.sum(x_map >= score_thr))
    top_row, top_col, top_score = top_candidate(x_map, score_thr)
    return x_map, candidate_count, top_row, top_col, top_score


def run_batch_dependence(model, dataset, batch_size, num_probes,
                         num_contexts, c, score_thr, seed):
    if batch_size < 1:
        raise ValueError('Configured test batch size must be positive.')
    probe_count = min(num_probes, len(dataset))
    probe_indices = list(range(probe_count))
    rows = []

    for probe_index in probe_indices:
        baseline = None
        baseline_count = None
        baseline_top = None
        for context_index in range(num_contexts):
            rng = random.Random(
                seed + probe_index * 100003 + context_index * 1009)
            indices = context_indices(
                len(dataset), probe_index, batch_size, rng)
            result = infer_probe_in_context(
                model, dataset, indices, c, score_thr)
            x_map, candidate_count, top_row, top_col, top_score = result

            if baseline is None:
                baseline = x_map.copy()
                baseline_count = candidate_count
                baseline_top = (top_row, top_col)
            difference = np.abs(x_map - baseline)
            top_changed = int((top_row, top_col) != baseline_top)
            rows.append({
                'probe_dataset_index': probe_index,
                'probe_file_id': str(dataset[probe_index]['file_id']),
                'context_index': context_index,
                'batch_size': batch_size,
                'max_abs_diff_vs_context0': float(difference.max()),
                'mean_abs_diff_vs_context0': float(difference.mean()),
                'baseline_candidate_count': baseline_count,
                'candidate_count': candidate_count,
                'candidate_count_changed': int(
                    candidate_count != baseline_count),
                'baseline_top_row': baseline_top[0],
                'baseline_top_col': baseline_top[1],
                'top_row': top_row,
                'top_col': top_col,
                'top_score': top_score,
                'top_position_changed': top_changed,
            })
    return rows


def write_batch_conclusion(path, rows):
    comparisons = [
        row for row in rows if row['context_index'] != 0
    ]
    tolerance = 1e-6
    changed_outputs = [
        row for row in comparisons
        if row['max_abs_diff_vs_context0'] > tolerance
    ]
    count_changes = sum(
        row['candidate_count_changed'] for row in comparisons)
    top_changes = sum(row['top_position_changed'] for row in comparisons)
    max_difference = max(
        (row['max_abs_diff_vs_context0'] for row in comparisons),
        default=0.0)
    mean_difference = np.mean([
        row['mean_abs_diff_vs_context0'] for row in comparisons
    ]) if comparisons else 0.0
    dependency_detected = bool(changed_outputs)

    lines = [
        'DynamicKernelMixture batch dependence diagnostic',
        f'comparisons={len(comparisons)}',
        f'numerical_change_tolerance={tolerance:g}',
        f'changed_output_comparisons={len(changed_outputs)}',
        f'changed_output_ratio='
        f'{len(changed_outputs) / len(comparisons) if comparisons else 0.0:.6f}',
        f'max_abs_diff_overall={max_difference:.9g}',
        f'mean_of_mean_abs_diff={mean_difference:.9g}',
        f'candidate_count_change_comparisons={count_changes}',
        f'top_position_change_comparisons={top_changes}',
        f'batch_dependence_detected={dependency_detected}',
        '',
        'Conclusion:',
    ]
    if dependency_detected:
        lines.append(
            'The same sample changed when only the other batch members were '
            'replaced. The current model output is batch-content dependent.')
    else:
        lines.append(
            'No batch-content dependence above the numerical tolerance was '
            'observed for the tested probes and contexts.')
    lines.append(
        'The probe always occupies batch slot 0 and the configured full batch '
        'size is preserved.')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.max_samples < 0:
        raise ValueError('--max-samples must be non-negative.')
    if args.num_probes < 1:
        raise ValueError('--num-probes must be at least 1.')
    if args.num_contexts < 2:
        raise ValueError('--num-contexts must be at least 2.')
    if args.match_distance < 0:
        raise ValueError('--match-distance must be non-negative.')

    cfg = Config.fromfile(args.config)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    c, score_thr, batch_size = resolve_options(cfg)
    device = select_device()
    set_model_module_device(device)

    dataloader = Runner.build_dataloader(cfg.test_dataloader)
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    sample_rows = []
    sample_index = 0
    with torch.no_grad():
        for data_batch in dataloader:
            output = unwrap_output(model(mode='predict', **data_batch))
            output_batch_size = int(output['x_final'].shape[0])
            if output_batch_size != batch_size:
                raise RuntimeError(
                    f'Expected full batch size {batch_size}, but model '
                    f'returned {output_batch_size}. Keep drop_last=True for '
                    'the current DynamicKernelMixture implementation.')
            for batch_index in range(output_batch_size):
                sample_rows.append(analyze_sample(
                    output,
                    batch_index,
                    sample_index,
                    c,
                    score_thr,
                    args.match_distance))
                sample_index += 1
                if args.max_samples and sample_index >= args.max_samples:
                    break
            if args.max_samples and sample_index >= args.max_samples:
                break

        batch_rows = run_batch_dependence(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            num_probes=args.num_probes,
            num_contexts=args.num_contexts,
            c=c,
            score_thr=score_thr,
            seed=args.seed)

    sample_csv = work_dir / 'per_sample_summary.csv'
    stats_txt = work_dir / 'overall_stats.txt'
    batch_csv = work_dir / 'batch_dependence_report.csv'
    batch_txt = work_dir / 'batch_dependence_conclusion.txt'
    write_csv(sample_csv, sample_rows)
    write_overall_stats(
        stats_txt, sample_rows, score_thr, args.match_distance)
    write_csv(batch_csv, batch_rows)
    write_batch_conclusion(batch_txt, batch_rows)

    print(f'device={device}')
    print(f'batch_size={batch_size}')
    print(f'analyzed_samples={len(sample_rows)}')
    print(f'output_dir={work_dir.resolve()}')
    print(f'per_sample_summary={sample_csv.resolve()}')
    print(f'overall_stats={stats_txt.resolve()}')
    print(f'batch_dependence_report={batch_csv.resolve()}')
    print(f'batch_dependence_conclusion={batch_txt.resolve()}')


if __name__ == '__main__':
    main()
