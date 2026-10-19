import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.runner.checkpoint import load_checkpoint

from arlr.datasets import *  # noqa: F401,F403
from arlr.evaluation import *  # noqa: F401,F403
from arlr.models import *  # noqa: F401,F403


AP_THRS = (0.05, 0.10, 0.15)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize success and failure modes of subpixel offsets.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val', choices=['val', 'test'])
    parser.add_argument('--max-samples', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--brightness-threshold', type=float, default=50.0)
    parser.add_argument('--c', type=int, default=3)
    parser.add_argument('--subpixel-offset-limit', type=float, default=0.5)
    parser.add_argument('--case-limit', type=int, default=24)
    parser.add_argument('--local-radius', type=int, default=4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def ensure_dirs(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
            'failure_cases', 'success_cases', 'crowded_cases',
            'anchor_conflict_cases'
    ]:
        (out_dir / name).mkdir(parents=True, exist_ok=True)


def build_batch(samples):
    keys = ['gt_img_11', 'gt', 'ann_path', 'file_id', 'count']
    return {key: [sample[key] for sample in samples] for key in keys}


def as_numpy_gt(ann):
    arr = np.asarray(ann, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return arr.reshape(-1, 3)


def pairwise_min_gt_distance(gt):
    if gt.shape[0] < 2:
        return math.inf
    pts = gt[:, :2]
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist += np.eye(dist.shape[0]) * 1e9
    return float(dist.min())


def nearest_gt(point, gt):
    if gt.shape[0] == 0:
        return -1, math.inf
    dist = np.linalg.norm(gt[:, :2] - point[None, :], axis=1)
    idx = int(np.argmin(dist))
    return idx, float(dist[idx])


def count_gt_within(point, gt, radius):
    if gt.shape[0] == 0:
        return 0
    dist = np.linalg.norm(gt[:, :2] - point[None, :], axis=1)
    return int(np.sum(dist <= radius))


def anchor_counts(gt, c):
    center = (c - 1) // 2
    counts = {}
    anchors = []
    for target in gt:
        anchor = (
            int(round(target[0] * c + center)),
            int(round(target[1] * c + center)),
        )
        anchors.append(anchor)
        counts[anchor] = counts.get(anchor, 0) + 1
    return anchors, counts


def is_local_max(matrix, row, col):
    row0 = max(row - 1, 0)
    row1 = min(row + 2, matrix.shape[0])
    col0 = max(col - 1, 0)
    col1 = min(col + 2, matrix.shape[1])
    return bool(matrix[row, col] >= np.max(matrix[row0:row1, col0:col1]))


def candidate_indices(matrix, brightness_threshold):
    thresholded = matrix.copy()
    thresholded[thresholded < brightness_threshold] = 0
    return np.nonzero(thresholded)


def infer_sample_records(args, sample_idx, batch_idx, output):
    center = (args.c - 1) // 2
    side = 11 * args.c
    matrix = output['x_final'][batch_idx].detach().cpu().numpy().reshape(
        side, side)
    offsets = output['subpixel_offset'][batch_idx].detach().cpu().numpy()
    gt = as_numpy_gt(output['targets_GT'][batch_idx])
    file_id = output['file_id'][batch_idx]
    gt_anchors, gt_anchor_counts = anchor_counts(gt, args.c)
    sample_min_gt_dist = pairwise_min_gt_distance(gt)

    rows, cols = candidate_indices(matrix, args.brightness_threshold)
    records = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        offset_row = float(np.clip(offsets[0, row, col],
                                  -args.subpixel_offset_limit,
                                  args.subpixel_offset_limit))
        offset_col = float(np.clip(offsets[1, row, col],
                                  -args.subpixel_offset_limit,
                                  args.subpixel_offset_limit))
        raw = np.asarray([(row - center) / args.c,
                          (col - center) / args.c], dtype=np.float64)
        refined = np.asarray([(row + offset_row - center) / args.c,
                              (col + offset_col - center) / args.c],
                             dtype=np.float64)
        raw_gt_idx, raw_dist = nearest_gt(raw, gt)
        refined_gt_idx, refined_dist = nearest_gt(refined, gt)
        nearest_anchor_count = 0
        if raw_gt_idx >= 0 and raw_gt_idx < len(gt_anchors):
            nearest_anchor_count = gt_anchor_counts[gt_anchors[raw_gt_idx]]
        delta = refined_dist - raw_dist

        record = {
            'sample_idx': sample_idx,
            'batch_idx': batch_idx,
            'file_id': file_id,
            'row': row,
            'col': col,
            'value': float(matrix[row, col]),
            'offset_row': offset_row,
            'offset_col': offset_col,
            'offset_norm': float(math.hypot(offset_row, offset_col)),
            'raw_row': float(raw[0]),
            'raw_col': float(raw[1]),
            'refined_row': float(refined[0]),
            'refined_col': float(refined[1]),
            'raw_nearest_gt_idx': raw_gt_idx,
            'refined_nearest_gt_idx': refined_gt_idx,
            'raw_dist': raw_dist,
            'refined_dist': refined_dist,
            'delta_dist': delta,
            'improved': int(delta < -1e-12),
            'worsened': int(delta > 1e-12),
            'unchanged': int(abs(delta) <= 1e-12),
            'local_max': int(is_local_max(matrix, row, col)),
            'gt_count': int(gt.shape[0]),
            'sample_min_gt_dist': sample_min_gt_dist,
            'raw_gt_within_0_25': count_gt_within(raw, gt, 0.25),
            'raw_gt_within_0_50': count_gt_within(raw, gt, 0.50),
            'raw_gt_within_1_00': count_gt_within(raw, gt, 1.00),
            'edge_dist_hr': min(row, col, side - 1 - row, side - 1 - col),
            'nearest_gt_anchor_count': int(nearest_anchor_count),
            'anchor_conflict': int(nearest_anchor_count > 1),
        }
        for thr in AP_THRS:
            key = f'ap{int(thr * 100):02d}'
            record[f'{key}_improved_in'] = int(raw_dist > thr and
                                               refined_dist <= thr)
            record[f'{key}_lost'] = int(raw_dist <= thr and refined_dist > thr)
        records.append(record)
    return records


def write_csv(path, records):
    if not records:
        path.write_text('', encoding='utf-8')
        return
    fieldnames = list(records[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def finite_array(records, key):
    values = [record[key] for record in records if np.isfinite(record[key])]
    return np.asarray(values, dtype=np.float64)


def save_overall_stats(path, records):
    lines = []
    lines.append(f'num_candidates={len(records)}')
    if not records:
        path.write_text('\n'.join(lines), encoding='utf-8')
        return

    raw = finite_array(records, 'raw_dist')
    refined = finite_array(records, 'refined_dist')
    delta = finite_array(records, 'delta_dist')
    offset_norm = finite_array(records, 'offset_norm')
    worsened = sum(record['worsened'] for record in records)
    improved = sum(record['improved'] for record in records)
    local_max = sum(record['local_max'] for record in records)
    anchor_conflict = sum(record['anchor_conflict'] for record in records)

    lines.extend([
        f'improved={improved}',
        f'worsened={worsened}',
        f'worsened_ratio={worsened / len(records):.6f}',
        f'local_max_ratio={local_max / len(records):.6f}',
        f'anchor_conflict_ratio={anchor_conflict / len(records):.6f}',
        f'raw_dist_mean={raw.mean():.6f}' if raw.size else 'raw_dist_mean=nan',
        (f'refined_dist_mean={refined.mean():.6f}'
         if refined.size else 'refined_dist_mean=nan'),
        (f'delta_dist_mean={delta.mean():.6f}'
         if delta.size else 'delta_dist_mean=nan'),
        (f'offset_norm_mean={offset_norm.mean():.6f}'
         if offset_norm.size else 'offset_norm_mean=nan'),
    ])
    for thr in AP_THRS:
        key = f'ap{int(thr * 100):02d}'
        improved_in = sum(record[f'{key}_improved_in'] for record in records)
        lost = sum(record[f'{key}_lost'] for record in records)
        raw_in = sum(record['raw_dist'] <= thr for record in records)
        refined_in = sum(record['refined_dist'] <= thr for record in records)
        lines.append(
            f'{key}: raw_in={raw_in}, refined_in={refined_in}, '
            f'improved_in={improved_in}, lost={lost}')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_hist_delta(path, records):
    delta = finite_array(records, 'delta_dist')
    if delta.size == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(delta, bins=80, color='#4C78A8', alpha=0.85)
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('refined distance - raw distance')
    plt.ylabel('candidate count')
    plt.title('Distance Change After Subpixel Offset')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_raw_vs_refined(path, records):
    raw = finite_array(records, 'raw_dist')
    refined = finite_array(records, 'refined_dist')
    if raw.size == 0 or refined.size == 0:
        return
    limit = max(np.percentile(raw, 99), np.percentile(refined, 99), 0.25)
    plt.figure(figsize=(6, 6))
    plt.scatter(raw, refined, s=8, alpha=0.35, c=np.asarray(refined) - raw,
                cmap='coolwarm', vmin=-0.15, vmax=0.15)
    plt.plot([0, limit], [0, limit], color='black', linestyle='--',
             linewidth=1)
    for thr in AP_THRS:
        plt.axvline(thr, color='gray', linestyle=':', linewidth=0.8)
        plt.axhline(thr, color='gray', linestyle=':', linewidth=0.8)
    plt.xlim(0, limit)
    plt.ylim(0, limit)
    plt.xlabel('raw nearest-GT distance')
    plt.ylabel('refined nearest-GT distance')
    plt.title('Raw vs Refined Localization Distance')
    plt.colorbar(label='refined - raw')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_offset_distribution(path, records, offset_limit):
    row = finite_array(records, 'offset_row')
    col = finite_array(records, 'offset_col')
    if row.size == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(row, bins=60, alpha=0.8, label='row')
    axes[0].hist(col, bins=60, alpha=0.65, label='col')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_title('Offset Marginals')
    axes[0].set_xlabel('high-res pixel offset')
    axes[0].legend()

    axes[1].scatter(col, row, s=6, alpha=0.25)
    axes[1].set_xlim(-offset_limit * 1.05, offset_limit * 1.05)
    axes[1].set_ylim(-offset_limit * 1.05, offset_limit * 1.05)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('col offset')
    axes[1].set_ylabel('row offset')
    axes[1].set_title('Offset Vector Distribution')

    norm = finite_array(records, 'offset_norm')
    axes[2].hist(norm, bins=60, color='#F58518', alpha=0.85)
    axes[2].set_title('Offset Norm')
    axes[2].set_xlabel('high-res pixel offset norm')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def bar_rate(ax, labels, totals, bads, title):
    rates = [(bad / total if total else 0.0)
             for bad, total in zip(bads, totals)]
    ax.bar(labels, rates, color='#E45756', alpha=0.85)
    for idx, (rate, total) in enumerate(zip(rates, totals)):
        ax.text(idx, rate + 0.01, f'n={total}', ha='center', fontsize=8)
    ax.set_ylim(0, max(rates + [0.05]) * 1.25)
    ax.set_ylabel('worsened ratio')
    ax.set_title(title)


def plot_failure_rates(path, records):
    if not records:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    labels = ['local max', 'non max']
    groups = [
        [r for r in records if r['local_max']],
        [r for r in records if not r['local_max']],
    ]
    bar_rate(axes[0, 0], labels, [len(g) for g in groups],
             [sum(r['worsened'] for r in g) for g in groups],
             'Failure Rate by Local-Max Status')

    labels = ['conflict', 'no conflict']
    groups = [
        [r for r in records if r['anchor_conflict']],
        [r for r in records if not r['anchor_conflict']],
    ]
    bar_rate(axes[0, 1], labels, [len(g) for g in groups],
             [sum(r['worsened'] for r in g) for g in groups],
             'Failure Rate by Anchor Conflict')

    bins = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0),
            (2.0, math.inf)]
    labels = ['0-.25', '.25-.5', '.5-1', '1-2', '>2']
    groups = []
    for low, high in bins:
        groups.append([
            r for r in records
            if np.isfinite(r['sample_min_gt_dist']) and
            r['sample_min_gt_dist'] >= low and r['sample_min_gt_dist'] < high
        ])
    bar_rate(axes[1, 0], labels, [len(g) for g in groups],
             [sum(r['worsened'] for r in g) for g in groups],
             'Failure Rate by GT Crowding')

    offset_norm = finite_array(records, 'offset_norm')
    if offset_norm.size:
        quantiles = np.quantile(offset_norm, [0, 0.25, 0.5, 0.75, 1.0])
        labels = ['Q1', 'Q2', 'Q3', 'Q4']
        groups = []
        for idx in range(4):
            low, high = quantiles[idx], quantiles[idx + 1]
            if idx == 3:
                groups.append([r for r in records
                               if r['offset_norm'] >= low and
                               r['offset_norm'] <= high])
            else:
                groups.append([r for r in records
                               if r['offset_norm'] >= low and
                               r['offset_norm'] < high])
        bar_rate(axes[1, 1], labels, [len(g) for g in groups],
                 [sum(r['worsened'] for r in g) for g in groups],
                 'Failure Rate by Offset Norm Quartile')
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def select_cases(records, case_limit):
    finite = [r for r in records if np.isfinite(r['raw_dist']) and
              np.isfinite(r['refined_dist'])]
    cases = {
        'failure_cases': sorted(
            [r for r in finite if r['worsened']],
            key=lambda r: r['delta_dist'],
            reverse=True)[:case_limit],
        'success_cases': sorted(
            [r for r in finite if r['ap10_improved_in'] or
             r['ap15_improved_in']],
            key=lambda r: r['raw_dist'] - r['refined_dist'],
            reverse=True)[:case_limit],
        'crowded_cases': sorted(
            [r for r in finite if np.isfinite(r['sample_min_gt_dist']) and
             r['sample_min_gt_dist'] <= 0.75],
            key=lambda r: r['delta_dist'],
            reverse=True)[:case_limit],
        'anchor_conflict_cases': sorted(
            [r for r in finite if r['anchor_conflict']],
            key=lambda r: r['delta_dist'],
            reverse=True)[:case_limit],
    }
    return cases


def run_one_sample(args, model, dataset, sample_idx):
    sample = dataset[sample_idx]
    with torch.no_grad():
        output = model(mode='predict', **build_batch([sample]))[0]
    return sample, output


def draw_case(path, args, sample, output, record):
    center = (args.c - 1) // 2
    side = 11 * args.c
    matrix = output['x_final'][0].detach().cpu().numpy().reshape(side, side)
    offsets = output['subpixel_offset'][0].detach().cpu().numpy()
    gt = as_numpy_gt(output['targets_GT'][0])
    raw_img = sample['gt_img_11'].view(11, 11).detach().cpu().numpy()

    rows, cols = candidate_indices(matrix, args.brightness_threshold)
    row, col = int(record['row']), int(record['col'])
    row_offset = float(np.clip(offsets[0, row, col],
                              -args.subpixel_offset_limit,
                              args.subpixel_offset_limit))
    col_offset = float(np.clip(offsets[1, row, col],
                              -args.subpixel_offset_limit,
                              args.subpixel_offset_limit))
    raw_hr = np.asarray([row, col], dtype=np.float64)
    refined_hr = np.asarray([row + row_offset, col + col_offset],
                            dtype=np.float64)
    gt_hr = gt[:, :2] * args.c + center if gt.size else np.zeros((0, 2))

    radius = args.local_radius
    row0 = max(row - radius, 0)
    row1 = min(row + radius + 1, side)
    col0 = max(col - radius, 0)
    col1 = min(col + radius + 1, side)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img, cmap='gray')
    axes[0].set_title('Input 11x11')
    axes[0].axis('off')

    axes[1].imshow(matrix, cmap='magma', origin='upper')
    axes[1].scatter(cols, rows, s=12, facecolors='none',
                    edgecolors='cyan', linewidths=0.6, label='raw candidates')
    if gt_hr.shape[0]:
        axes[1].scatter(gt_hr[:, 1], gt_hr[:, 0], marker='*', s=110,
                        color='lime', edgecolors='black', linewidths=0.5,
                        label='GT')
    axes[1].scatter([raw_hr[1]], [raw_hr[0]], s=70, facecolors='none',
                    edgecolors='white', linewidths=1.6, label='selected raw')
    axes[1].scatter([refined_hr[1]], [refined_hr[0]], s=70, color='red',
                    label='selected refined')
    axes[1].arrow(raw_hr[1], raw_hr[0], col_offset, row_offset,
                  color='white', width=0.015, head_width=0.35,
                  length_includes_head=True)
    axes[1].set_title('Full 33x33 Prediction')
    axes[1].legend(fontsize=7, loc='upper right')

    local = matrix[row0:row1, col0:col1]
    axes[2].imshow(local, cmap='magma', origin='upper',
                   extent=[col0 - 0.5, col1 - 0.5, row1 - 0.5, row0 - 0.5])
    if gt_hr.shape[0]:
        inside = ((gt_hr[:, 0] >= row0 - 0.5) &
                  (gt_hr[:, 0] <= row1 - 0.5) &
                  (gt_hr[:, 1] >= col0 - 0.5) &
                  (gt_hr[:, 1] <= col1 - 0.5))
        axes[2].scatter(gt_hr[inside, 1], gt_hr[inside, 0], marker='*',
                        s=140, color='lime', edgecolors='black',
                        linewidths=0.5)
    axes[2].scatter([raw_hr[1]], [raw_hr[0]], s=80, facecolors='none',
                    edgecolors='white', linewidths=1.8)
    axes[2].scatter([refined_hr[1]], [refined_hr[0]], s=80, color='red')
    axes[2].arrow(raw_hr[1], raw_hr[0], col_offset, row_offset,
                  color='white', width=0.01, head_width=0.25,
                  length_includes_head=True)
    axes[2].set_xlim(col0 - 0.5, col1 - 0.5)
    axes[2].set_ylim(row1 - 0.5, row0 - 0.5)
    axes[2].grid(color='white', alpha=0.25, linewidth=0.5)
    axes[2].set_title('Local Offset Geometry')

    title = (
        f'file={record["file_id"]} row={row} col={col} '
        f'value={record["value"]:.2f}\n'
        f'offset=({row_offset:.3f},{col_offset:.3f}) '
        f'raw={record["raw_dist"]:.4f} '
        f'refined={record["refined_dist"]:.4f} '
        f'delta={record["delta_dist"]:.4f} '
        f'local_max={record["local_max"]} '
        f'gt_count={record["gt_count"]} '
        f'min_gt_dist={record["sample_min_gt_dist"]:.3f} '
        f'anchor_conflict={record["anchor_conflict"]}')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_cases(args, model, dataset, out_dir, case_groups):
    for group_name, records in case_groups.items():
        group_dir = out_dir / group_name
        for idx, record in enumerate(records):
            sample, output = run_one_sample(args, model, dataset,
                                            int(record['sample_idx']))
            filename = (
                f'{idx:03d}_sample{record["sample_idx"]}_'
                f'file{record["file_id"]}_r{record["row"]}_'
                f'c{record["col"]}_delta{record["delta_dist"]:.4f}.png')
            draw_case(group_dir / filename, args, sample, output, record)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dirs(out_dir)

    cfg = Config.fromfile(args.config)
    dataset_cfg = (cfg.val_dataloader.dataset if args.split == 'val'
                   else cfg.test_dataloader.dataset)
    dataset = DATASETS.build(dataset_cfg)

    device = torch.device(args.device if torch.cuda.is_available() and
                          args.device.startswith('cuda') else 'cpu')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    max_samples = min(args.max_samples, len(dataset))
    records = []
    with torch.no_grad():
        for start in range(0, max_samples, args.batch_size):
            end = min(start + args.batch_size, max_samples)
            samples = [dataset[idx] for idx in range(start, end)]
            output = model(mode='predict', **build_batch(samples))[0]
            for batch_idx, sample_idx in enumerate(range(start, end)):
                records.extend(infer_sample_records(
                    args, sample_idx, batch_idx, output))

    write_csv(out_dir / 'summary.csv', records)
    save_overall_stats(out_dir / 'overall_stats.txt', records)
    plot_hist_delta(out_dir / 'hist_delta_distance.png', records)
    plot_raw_vs_refined(out_dir / 'scatter_raw_vs_refined.png', records)
    plot_offset_distribution(out_dir / 'offset_distribution.png', records,
                             args.subpixel_offset_limit)
    plot_failure_rates(out_dir / 'failure_rate_breakdown.png', records)

    case_groups = select_cases(records, args.case_limit)
    for group_name, group_records in case_groups.items():
        write_csv(out_dir / group_name / 'cases.csv', group_records)
    save_cases(args, model, dataset, out_dir, case_groups)


if __name__ == '__main__':
    main()
