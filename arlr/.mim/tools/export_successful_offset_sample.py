"""Export analysis and paper views of a successful sub-pixel offset case.

Candidate extraction and matching intentionally mirror CSO_Metrics:
all x_final grid cells at or above ``brightness_threshold`` are candidates;
there is no local-maximum suppression in CSO_Metrics.  Candidates are sorted
by intensity and matched one-to-one to their nearest GT within a distance
threshold, exactly as CSO_tpfp does.
"""

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.runner.checkpoint import load_checkpoint
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arlr.datasets import *  # noqa: E402,F401,F403
from arlr.evaluation import *  # noqa: E402,F401,F403
from arlr.models import *  # noqa: E402,F401,F403


GT_COLOR = '#FFD34E'
PRED_COLOR = '#FF4D5A'
REFINED_PRED_COLOR = '#35D4E8'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find and export one sample-level successful offset case.')
    parser.add_argument('--config', required=True, help='Model config file.')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint.')
    parser.add_argument('--out-dir', required=True, help='Output directory.')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Number of samples to search (default: all).')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default=None,
                        help='cuda, cuda:0, or cpu (default: cuda if available).')
    parser.add_argument('--brightness-threshold', type=float, default=None,
                        help='CSO candidate threshold (default: config value).')
    parser.add_argument('--subpixel-offset-limit', type=float, default=None,
                        help='CSO offset clamp limit (default: config value).')
    parser.add_argument('--match-distance-threshold', type=float, default=0.10,
                        help='CSO one-to-one match distance in LR coordinates.')
    parser.add_argument('--min-raw-distance', type=float, default=0.10,
                        help='The most improved pair must start at least this far away.')
    parser.add_argument('--min-improvement', type=float, default=0.08,
                        help='The most improved pair must improve by at least this much.')
    parser.add_argument('--min-sample-improvement', type=float, default=0.03,
                        help='Mean matched-pair improvement considered clearly positive.')
    parser.add_argument('--min-improved-ratio', type=float, default=0.50,
                        help='Fraction of refined matched targets that must improve '
                             'when the mean improvement is not already clear.')
    parser.add_argument('--min-refined-matches', type=int, default=1,
                        help='Minimum refined one-to-one matches in a selected sample.')
    parser.add_argument('--min-refined-precision', type=float, default=0.50,
                        help='Minimum refined sample precision under CSO matching.')
    parser.add_argument('--allow-relaxed-fallback', action='store_true',
                        help='Only if no default-success case exists, allow a '
                             'smaller positive average improvement. Disabled by default.')
    parser.add_argument('--zoom-pad', type=float, default=0.35,
                        help='LR-coordinate padding around the focal offset pair.')
    parser.add_argument('--zoom-min-size', type=float, default=1.10,
                        help='Minimum side length of the local zoom in LR pixels.')
    parser.add_argument('--localization-layout',
                        choices=('framework', 'analysis'),
                        default='framework',
                        help='framework: clean square panels for an architecture '
                             'diagram (default); analysis: full-view plus zoom '
                             'with titles and legend.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='PNG resolution; each panel is 4 inches square.')
    return parser.parse_args()


def get_metric_cfg(cfg):
    evaluator = cfg.get('test_evaluator', cfg.get('val_evaluator', {}))
    if isinstance(evaluator, (list, tuple)):
        for item in evaluator:
            if item.get('type') == 'CSO_Metrics':
                return item
        return evaluator[0] if evaluator else {}
    return evaluator or {}


def resolve_device(requested):
    if requested is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device requests CUDA, but CUDA is unavailable.')
    return device


def prepare_model(cfg, checkpoint, device):
    # ARLR.forward reads a module-level device and keeps Phi/Qinit as plain
    # tensor attributes, so this mirrors the project's comparison visualizer.
    if cfg.model.get('type') == 'ARLR':
        arlr_module = importlib.import_module('arlr.models.editor.arlr')
        arlr_module.device = device
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location=device)
    model.to(device)
    for name in ('Phi', 'Qinit'):
        value = getattr(model, name, None)
        if torch.is_tensor(value):
            setattr(model, name, value.to(device))
    model.eval()
    return model


def make_batch(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def as_gt_array(targets):
    gt = np.asarray(targets, dtype=np.float64)
    return gt.reshape(-1, 3) if gt.size else np.zeros((0, 3), dtype=np.float64)


def extract_prediction_maps(output, c):
    """Validate the ARLR predict tensors before using their coordinates."""
    if 'x_final' not in output:
        raise KeyError('ARLR predict output has no x_final.')
    x_final = output['x_final'].detach().float().cpu()
    if x_final.ndim != 2:
        raise ValueError(f'x_final must be [B, N], got {tuple(x_final.shape)}.')
    side = int(round(math.sqrt(x_final.shape[1])))
    expected_side = 11 * c
    if side * side != x_final.shape[1] or side != expected_side:
        raise ValueError(
            f'x_final must be [B, {(11 * c) ** 2}] for c={c}; got '
            f'{tuple(x_final.shape)}.')
    if 'subpixel_offset' not in output:
        raise KeyError('Offset-enabled evaluation requires subpixel_offset.')
    offsets = output['subpixel_offset'].detach().float().cpu()
    expected_shape = (x_final.shape[0], 2, side, side)
    if tuple(offsets.shape) != expected_shape:
        raise ValueError(
            f'subpixel_offset must be {expected_shape}, got '
            f'{tuple(offsets.shape)}.')
    return x_final.numpy().reshape(-1, side, side), offsets.numpy()


def detections_from_map(matrix, c, brightness_threshold, offset_limit,
                        subpixel_offset=None):
    """Replicate CSO_Metrics.process candidate and coordinate logic exactly.

    ``subpixel_offset[0]`` is the row displacement and ``[1]`` the column
    displacement. SubpixelOffsetHead has already applied tanh and its scale in
    the model; CSO_Metrics nevertheless clips each channel to offset_limit.
    The offsets are added in the 33x33 HR grid before conversion to LR coords.
    """
    filtered = np.asarray(matrix, dtype=np.float32).copy()
    filtered[filtered < brightness_threshold] = 0
    center = (c - 1) // 2
    rows, cols = np.nonzero(filtered)
    detections = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        row_offset = 0.0
        col_offset = 0.0
        if subpixel_offset is not None:
            row_offset = float(np.clip(subpixel_offset[0, row, col],
                                       -offset_limit, offset_limit))
            col_offset = float(np.clip(subpixel_offset[1, row, col],
                                       -offset_limit, offset_limit))
        detections.append({
            'row': float((row + row_offset - center) / c),
            'col': float((col + col_offset - center) / c),
            'score': float(filtered[row, col]),
            'grid_row': int(row),
            'grid_col': int(col),
            'offset_row_hr': row_offset,
            'offset_col_hr': col_offset,
        })
    detections.sort(key=lambda item: item['score'], reverse=True)
    return detections, filtered


def cso_one_to_one_matches(gt, detections, distance_threshold):
    """Return (GT index, detection index, distance), matching CSO_tpfp.

    CSO_tpfp first finds each detection's nearest GT, sorts detections by
    brightness, then lets a GT be used only once.  This is deliberately not a
    global nearest-pair assignment, because the visualized matching must agree
    with the evaluator.
    """
    if len(gt) == 0 or not detections:
        return []
    det_rows_cols = np.asarray(
        [[det['row'], det['col']] for det in detections], dtype=np.float64)
    distances = np.linalg.norm(
        det_rows_cols[:, None, :] - gt[None, :, :2], axis=2)
    nearest_gt = distances.argmin(axis=1)
    nearest_distance = distances[np.arange(len(detections)), nearest_gt]
    order = np.argsort(-np.asarray([det['score'] for det in detections]))
    used_gt = set()
    matches = []
    for det_index in order.tolist():
        gt_index = int(nearest_gt[det_index])
        distance = float(nearest_distance[det_index])
        if distance <= distance_threshold and gt_index not in used_gt:
            used_gt.add(gt_index)
            matches.append((gt_index, int(det_index), distance))
    return matches


def match_summary(gt, detections, matches):
    distances = [item[2] for item in matches]
    return {
        'gt_count': int(len(gt)),
        'candidate_count': int(len(detections)),
        'matched_count': int(len(matches)),
        'recall': float(len(matches) / len(gt)) if len(gt) else 0.0,
        'precision': float(len(matches) / len(detections)) if detections else 0.0,
        'mean_matched_distance': float(np.mean(distances)) if distances else float('inf'),
        'max_matched_distance': float(np.max(distances)) if distances else float('inf'),
    }


def serializable_matches(matches, gt, detections):
    result = []
    for gt_index, det_index, distance in matches:
        det = detections[det_index]
        result.append({
            'gt_index': int(gt_index),
            'prediction_index': int(det_index),
            'distance': float(distance),
            'gt_row': float(gt[gt_index, 0]),
            'gt_col': float(gt[gt_index, 1]),
            'prediction_row': float(det['row']),
            'prediction_col': float(det['col']),
            'grid_row': int(det['grid_row']),
            'grid_col': int(det['grid_col']),
            'score': float(det['score']),
        })
    return result


def sample_record(matrix, offsets, gt, c, threshold, offset_limit,
                  match_distance_threshold):
    raw_detections, filtered = detections_from_map(
        matrix, c, threshold, offset_limit, subpixel_offset=None)
    refined_detections, _ = detections_from_map(
        matrix, c, threshold, offset_limit, subpixel_offset=offsets)
    raw_matches = cso_one_to_one_matches(
        gt, raw_detections, match_distance_threshold)
    refined_matches = cso_one_to_one_matches(
        gt, refined_detections, match_distance_threshold)

    raw_by_grid = {(det['grid_row'], det['grid_col']): det
                   for det in raw_detections}
    comparisons = []
    for gt_index, refined_index, refined_distance in refined_matches:
        refined = refined_detections[refined_index]
        raw = raw_by_grid[(refined['grid_row'], refined['grid_col'])]
        raw_distance = float(math.hypot(
            raw['row'] - gt[gt_index, 0], raw['col'] - gt[gt_index, 1]))
        comparisons.append({
            'gt_index': int(gt_index),
            'refined_prediction_index': int(refined_index),
            'grid_row': int(refined['grid_row']),
            'grid_col': int(refined['grid_col']),
            'raw_distance_to_matched_gt': raw_distance,
            'refined_distance_to_matched_gt': float(refined_distance),
            'improvement': raw_distance - float(refined_distance),
            'offset_row_hr': float(refined['offset_row_hr']),
            'offset_col_hr': float(refined['offset_col_hr']),
        })
    focal = max(comparisons, key=lambda item: item['improvement']) if comparisons else None
    improvements = [item['improvement'] for item in comparisons]
    improved_count = int(sum(value > 1e-9 for value in improvements))
    worsened_count = int(sum(value < -1e-9 for value in improvements))
    return {
        'raw_detections': raw_detections,
        'refined_detections': refined_detections,
        'raw_matches': raw_matches,
        'refined_matches': refined_matches,
        'raw_summary': match_summary(gt, raw_detections, raw_matches),
        'refined_summary': match_summary(gt, refined_detections, refined_matches),
        'comparisons': comparisons,
        'mean_matched_improvement': (float(np.mean(improvements))
                                     if improvements else float('-inf')),
        'improved_match_count': improved_count,
        'worsened_match_count': worsened_count,
        'improved_match_ratio': (float(improved_count / len(improvements))
                                 if improvements else 0.0),
        'worsened_match_ratio': (float(worsened_count / len(improvements))
                                 if improvements else 0.0),
        'focal_pair': focal,
        'filtered_matrix': filtered,
    }


def is_strict_success(record, args):
    refined = record['refined_summary']
    focal = record['focal_pair']
    overall_better = record['mean_matched_improvement'] > 1e-9
    majority_or_clear_mean = (
        record['improved_match_ratio'] >= args.min_improved_ratio or
        record['mean_matched_improvement'] >= args.min_sample_improvement)
    return (
        focal is not None and
        refined['matched_count'] >= args.min_refined_matches and
        refined['precision'] >= args.min_refined_precision and
        overall_better and
        majority_or_clear_mean and
        focal['raw_distance_to_matched_gt'] >= args.min_raw_distance and
        focal['improvement'] >= args.min_improvement)


def is_relaxed_success(record, args):
    """Opt-in fallback: still positive overall, but less representative."""
    refined = record['refined_summary']
    focal = record['focal_pair']
    return (
        focal is not None and
        refined['matched_count'] >= 1 and
        refined['precision'] >= min(args.min_refined_precision, 0.35) and
        record['mean_matched_improvement'] > 1e-9 and
        focal['improvement'] >= max(args.min_improvement, 0.04))


def success_rank(record):
    focal = record['focal_pair']
    refined = record['refined_summary']
    return (
        record['mean_matched_improvement'],
        record['improved_match_ratio'],
        focal['improvement'],
        refined['precision'],
        refined['recall'],
    )


def find_best_sample(args, model, dataset, c, threshold, offset_limit):
    max_samples = len(dataset) if args.max_samples is None else min(
        args.max_samples, len(dataset))
    strict_best = None
    relaxed_best = None
    with torch.no_grad():
        for start in range(0, max_samples, args.batch_size):
            end = min(start + args.batch_size, max_samples)
            samples = [dataset[index] for index in range(start, end)]
            output = model(mode='predict', **make_batch(samples))[0]
            matrices, offsets = extract_prediction_maps(output, c)
            for batch_index, sample_index in enumerate(range(start, end)):
                gt = as_gt_array(output['targets_GT'][batch_index])
                if not len(gt):
                    continue
                record = sample_record(
                    matrices[batch_index], offsets[batch_index], gt, c,
                    threshold, offset_limit, args.match_distance_threshold)
                record['sample_index'] = int(sample_index)
                record['file_id'] = str(output['file_id'][batch_index])
                if is_strict_success(record, args) and (
                        strict_best is None or success_rank(record) > success_rank(strict_best)):
                    strict_best = record
                if args.allow_relaxed_fallback and is_relaxed_success(record, args) and (
                        relaxed_best is None or success_rank(record) > success_rank(relaxed_best)):
                    relaxed_best = record
    if strict_best is not None:
        return strict_best, 'strict'
    if relaxed_best is not None:
        return relaxed_best, 'explicit_relaxed_fallback'
    raise RuntimeError(
        'No sample had a positive overall offset improvement and a representative '
        'corrected target under the requested criteria. No fallback is used by '
        'default; search more samples or explicitly pass '
        '--allow-relaxed-fallback.')


def normalize_uint8(matrix):
    image = np.nan_to_num(np.maximum(matrix, 0), nan=0.0)
    peak = float(image.max())
    if peak > 0:
        image = image / peak
    return np.uint8(np.round(image * 255))


def save_original_input(path, raw_input, dpi):
    image = Image.fromarray(normalize_uint8(raw_input), mode='L')
    target = max(image.width, int(round(4 * dpi)))
    resampling = getattr(Image, 'Resampling', Image)
    image.resize((target, target), resampling.BICUBIC).save(path, dpi=(dpi, dpi))


def save_intensity_map(path, matrix, dpi):
    image = Image.fromarray(normalize_uint8(matrix), mode='L')
    target = max(image.width, int(round(4 * dpi)))
    resampling = getattr(Image, 'Resampling', Image)
    image.resize((target, target), resampling.NEAREST).save(path, dpi=(dpi, dpi))


def zoom_bounds(gt, record, after_offset, args, c):
    """Return HR-grid (xmin, xmax, ymin, ymax) around the focal offset."""
    focal = record['focal_pair']
    gt_point = gt[focal['gt_index'], :2]
    raw = next(det for det in record['raw_detections']
               if (det['grid_row'], det['grid_col']) ==
               (focal['grid_row'], focal['grid_col']))
    refined = next(det for det in record['refined_detections']
                   if (det['grid_row'], det['grid_col']) ==
                   (focal['grid_row'], focal['grid_col']))
    center = (c - 1) // 2
    points = [gt_point * c + center,
              [raw['grid_row'], raw['grid_col']]]
    if after_offset:
        points.append([refined['grid_row'] + refined['offset_row_hr'],
                       refined['grid_col'] + refined['offset_col_hr']])
    points = np.asarray(points, dtype=np.float64)
    row_min, col_min = points.min(axis=0)
    row_max, col_max = points.max(axis=0)
    size = max(row_max - row_min + 2 * args.zoom_pad * c,
               col_max - col_min + 2 * args.zoom_pad * c,
               args.zoom_min_size * c)
    row_center = (row_min + row_max) / 2.0
    col_center = (col_min + col_max) / 2.0
    return (col_center - size / 2.0, col_center + size / 2.0,
            row_center - size / 2.0, row_center + size / 2.0)


def draw_axis(ax, matrix, gt, detections, matches, c, title, zoom=None,
              draw_offset_arrows=False, raw_detections=None):
    """Plot all CSO candidates in continuous HR-grid coordinates.

    Match lines identify the evaluator's one-to-one assignments; prediction
    circles intentionally include unmatched candidates too, so the complete
    sample-level detection result is visible in both panels. Raw predictions
    are integer HR-grid indices; GT and offset-refined predictions can be
    fractional. Evaluator matching itself remains in LR coordinates.
    """
    side = matrix.shape[0]
    center = (c - 1) // 2
    display = np.ma.masked_where(matrix <= 0, matrix)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')
    ax.set_facecolor('black')
    ax.imshow(display, cmap=cmap, origin='upper', interpolation='nearest',
              extent=(-0.5, side - 0.5, side - 0.5, -0.5), alpha=0.86)

    # Ground truth: yellow x. Before offset: red hollow circle.
    gt_rows = gt[:, 0] * c + center
    gt_cols = gt[:, 1] * c + center
    ax.scatter(gt_cols, gt_rows, marker='x', color=GT_COLOR, s=140,
               linewidths=2.5, label='Ground truth', zorder=6)
    # Use grid index + learned HR-pixel offset directly. With zero offset this
    # is exactly an integer grid point, without an LR round trip.
    det_rows = [det['grid_row'] + det['offset_row_hr'] for det in detections]
    det_cols = [det['grid_col'] + det['offset_col_hr'] for det in detections]
    if draw_offset_arrows and raw_detections is not None:
        # The after-offset panel must show both states: the red circle is the
        # original grid prediction and the blue circle is its refined location.
        # Every arrow therefore has an explicitly visible start and end.
        raw_rows = [det['grid_row'] for det in raw_detections]
        raw_cols = [det['grid_col'] for det in raw_detections]
        ax.scatter(raw_cols, raw_rows, marker='o',
                   facecolors='none', edgecolors=PRED_COLOR, s=112,
                   linewidths=2.2, label='Original grid prediction', zorder=7)
        if detections:
            ax.scatter(det_cols, det_rows, marker='o',
                       facecolors='none', edgecolors=REFINED_PRED_COLOR,
                       s=88, linewidths=2.0, label='Offset prediction',
                       zorder=8)
    elif detections:
        ax.scatter(det_cols, det_rows, marker='o',
                   facecolors='none', edgecolors=PRED_COLOR, s=100,
                   linewidths=2.0, label='Original grid prediction', zorder=7)
    for gt_index, det_index, _ in matches:
        ax.plot([gt_cols[gt_index], det_cols[det_index]],
                [gt_rows[gt_index], det_rows[det_index]],
                color='#DDE7EA', linewidth=0.8, alpha=0.72, zorder=4)
    if draw_offset_arrows and raw_detections is not None:
        raw_by_grid = {(det['grid_row'], det['grid_col']): det
                       for det in raw_detections}
        for _, det_index, _ in matches:
            refined = detections[det_index]
            raw = raw_by_grid[(refined['grid_row'], refined['grid_col'])]
            refined_row = refined['grid_row'] + refined['offset_row_hr']
            refined_col = refined['grid_col'] + refined['offset_col_hr']
            ax.annotate('', xy=(refined_col, refined_row),
                        xytext=(raw['grid_col'], raw['grid_row']),
                        arrowprops=dict(arrowstyle='->', color=REFINED_PRED_COLOR,
                                        lw=1.6, shrinkA=7, shrinkB=6),
                        zorder=9)
    if zoom is None:
        ax.set_xlim(-0.5, side - 0.5)
        ax.set_ylim(side - 0.5, -0.5)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(5))
    else:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[3], zoom[2])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
    # Integer HR coordinates are pixel centers. Draw cell boundaries only at
    # half-integers so raw predictions visibly sit in the middle of a cell.
    cell_boundaries = np.arange(-0.5, side + 0.5, 1.0)
    ax.xaxis.set_minor_locator(FixedLocator(cell_boundaries))
    ax.yaxis.set_minor_locator(FixedLocator(cell_boundaries))
    ax.set_axisbelow(False)
    ax.grid(which='minor', color='#91A6AE', linewidth=0.75, alpha=0.50,
            zorder=3)
    # Locators must not be allowed to expand a requested local crop.
    if zoom is None:
        ax.set_xlim(-0.5, side - 0.5)
        ax.set_ylim(side - 0.5, -0.5)
    else:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[3], zoom[2])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=10, color='white', pad=7)
    ax.set_xlabel('HR grid column index', color='#C8D4D8', fontsize=8)
    ax.set_ylabel('HR grid row index', color='#C8D4D8', fontsize=8)
    ax.tick_params(colors='#C8D4D8', labelsize=8)
    ax.tick_params(which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_color('#829197')


def save_analysis_localization_view(path, record, gt, c, after_offset, args):
    """Save a full-view + local-zoom composite for detailed analysis."""
    detections = (record['refined_detections'] if after_offset
                  else record['raw_detections'])
    matches = record['refined_matches'] if after_offset else record['raw_matches']
    stage = 'After offset' if after_offset else 'Before offset'
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=args.dpi,
                                facecolor='black')
    draw_axis(axes[0], record['filtered_matrix'], gt, detections, matches, c,
              f'{stage}: full view', draw_offset_arrows=after_offset,
              raw_detections=record['raw_detections'])
    draw_axis(axes[1], record['filtered_matrix'], gt, detections, matches, c,
              f'{stage}: local zoom',
              zoom=zoom_bounds(gt, record, after_offset, args, c),
              draw_offset_arrows=after_offset,
              raw_detections=record['raw_detections'])
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(handles, labels, loc='lower center', ncol=3,
                               frameon=False, fontsize=9,
                               bbox_to_anchor=(0.5, 0.01))
        for text in legend.get_texts():
            text.set_color('white')
    figure.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.20,
                            wspace=0.14)
    figure.savefig(path, dpi=args.dpi, facecolor='black')
    plt.close(figure)


def save_offset_local_comparison(path, record, gt, c, args):
    """Place the before/after local zooms side by side on one shared scale."""
    zoom = zoom_bounds(gt, record, after_offset=True, args=args, c=c)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=args.dpi,
                                facecolor='black')
    draw_axis(axes[0], record['filtered_matrix'], gt,
              record['raw_detections'], record['raw_matches'], c,
              'Before offset: HR-grid zoom', zoom=zoom)
    draw_axis(axes[1], record['filtered_matrix'], gt,
              record['refined_detections'], record['refined_matches'], c,
              'After offset: HR-grid zoom', zoom=zoom,
              draw_offset_arrows=True,
              raw_detections=record['raw_detections'])
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        legend = figure.legend(handles, labels, loc='lower center', ncol=3,
                               frameon=False, fontsize=9,
                               bbox_to_anchor=(0.5, 0.01))
        for text in legend.get_texts():
            text.set_color('white')
    figure.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.20,
                            wspace=0.16)
    figure.savefig(path, dpi=args.dpi, facecolor='black')
    plt.close(figure)


def _paper_intensity_axis(ax, matrix, cmap_name, vmax=None):
    """Draw a reconstruction panel with paper-friendly outer styling."""
    image = np.asarray(matrix, dtype=np.float32).copy()
    image[~np.isfinite(image)] = 0.0
    image = np.maximum(image, 0.0)
    positive = image[image > 0]
    if vmax is None:
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-6)
    side = image.shape[0]
    ax.set_facecolor('#090B10')
    ax.imshow(image, cmap=cmap_name, vmin=0.0, vmax=vmax,
              interpolation='nearest', origin='upper',
              extent=(-0.5, side - 0.5, side - 0.5, -0.5))
    ax.set_xlim(-0.5, side - 0.5)
    ax.set_ylim(side - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#6F7882')
        spine.set_linewidth(0.8)
    return vmax


def _add_figure_arrow(figure, source_ax, target_ax, label):
    """Connect adjacent panels without consuming image-space annotations."""
    source = source_ax.get_position()
    target = target_ax.get_position()
    y = (source.y0 + source.y1) / 2.0
    start = (source.x1 + 0.005, y)
    end = (target.x0 - 0.005, y)
    arrow = FancyArrowPatch(
        start, end, transform=figure.transFigure, arrowstyle='-|>',
        mutation_scale=10, linewidth=1.0, color='#000000', zorder=20)
    figure.add_artist(arrow)
    figure.text((start[0] + end[0]) / 2.0, y + 0.018, label,
                ha='center', va='bottom', fontsize=7.2,
                fontweight='normal', color='#000000')


def save_intro_motivation_figure(path, raw_input, record, gt, c, args):
    """Export a compact three-panel motivation figure for the introduction.

    The first two panels establish the observation-to-unmixing context.
    The final panel enlarges one real matched target and makes the core issue
    explicit: a grid prediction is restricted to an integer cell center while
    the target coordinate is continuous.
    """
    focal = record['focal_pair']
    if focal is None:
        raise ValueError('An introduction figure requires a focal matched pair.')
    raw = next(det for det in record['raw_detections']
               if (det['grid_row'], det['grid_col']) ==
               (focal['grid_row'], focal['grid_col']))
    center = (c - 1) // 2
    gt_row, gt_col = gt[focal['gt_index'], :2] * c + center
    raw_row, raw_col = float(raw['grid_row']), float(raw['grid_col'])
    zoom = zoom_bounds(gt, record, after_offset=True, args=args, c=c)
    candidate_map = record['filtered_matrix']

    figure, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), dpi=args.dpi,
                                facecolor='white')
    figure.subplots_adjust(left=0.035, right=0.975, top=0.84, bottom=0.10,
                           wspace=0.28)

    _paper_intensity_axis(axes[0], raw_input, 'gray')
    # The introduction should show the sparse HR localization evidence, not
    # every low-amplitude value of the raw reconstruction.  Reuse the exact
    # brightness-thresholded map used to create CSO candidates and render it
    # in grayscale so active HR cells read as a few white points.
    candidate_vmax = _paper_intensity_axis(axes[1], candidate_map, 'gray')
    # The explanatory zoom is intentionally schematic.  Showing one enlarged
    # white HR pixel as a filled cell makes the panel look like a white block
    # and distracts from the coordinate mismatch, so retain only the grid,
    # the discrete prediction, and the continuous target location here.
    _paper_intensity_axis(axes[2], np.zeros_like(candidate_map), 'gray',
                          vmax=candidate_vmax)

    titles = (
        '(a) Mixed infrared observation',
        '(b) Grid-based target recovery',
        '(c) Discrete vs. continuous localization',
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=8.8, fontweight='normal',
                     color='#000000', pad=8)

    # The grid-based recovery establishes where the local evidence comes from.
    zoom_box = Rectangle(
        (zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2],
        fill=False, edgecolor=REFINED_PRED_COLOR, linewidth=1.4,
        linestyle=(0, (3, 2)), zorder=8)
    axes[1].add_patch(zoom_box)
    # Preserve cell geometry in the local panel: integer coordinates are cell
    # centers and half-integers are boundaries.
    local_ax = axes[2]
    side = candidate_map.shape[0]
    boundaries = np.arange(-0.5, side + 0.5, 1.0)
    local_ax.xaxis.set_minor_locator(FixedLocator(boundaries))
    local_ax.yaxis.set_minor_locator(FixedLocator(boundaries))
    local_ax.grid(which='minor', color='#AEB9C2', linewidth=0.8,
                  alpha=0.72, zorder=4)
    local_ax.tick_params(which='minor', length=0)
    local_ax.set_xlim(zoom[0], zoom[1])
    local_ax.set_ylim(zoom[3], zoom[2])

    # The dashed straight arrow is the unavoidable grid quantization gap.
    local_ax.annotate(
        '', xy=(gt_col, gt_row), xytext=(raw_col, raw_row),
        arrowprops=dict(arrowstyle='->', color='#FF0000', linewidth=1.7,
                        shrinkA=7, shrinkB=8),
        zorder=9)
    local_ax.text(
        0.50, 0.13, 'Grid quantization error',
        transform=local_ax.transAxes, ha='center', va='center',
        fontsize=7.1, fontweight='normal', color='#FF0000',
        bbox=dict(boxstyle='round,pad=0.22', facecolor='#090B10',
                  edgecolor=PRED_COLOR, linewidth=0.7, alpha=0.94),
        zorder=12)

    local_ax.scatter([raw_col], [raw_row], marker='o', facecolors='none',
                     edgecolors=PRED_COLOR, s=74, linewidths=2.0, zorder=13)
    local_ax.scatter([gt_col], [gt_row], marker='x', color=GT_COLOR, s=82,
                     linewidths=2.3, zorder=15)

    _add_figure_arrow(figure, axes[0], axes[1], 'unmixing')
    _add_figure_arrow(figure, axes[1], axes[2], 'zoom')

    # Trim the now-visible white outer canvas while retaining enough breathing
    # room that titles and panel borders do not touch the exported page edge.
    figure.savefig(path, dpi=args.dpi, facecolor='white',
                   bbox_inches='tight', pad_inches=0.08)
    figure.savefig(Path(path).with_suffix('.pdf'), facecolor='white',
                   bbox_inches='tight', pad_inches=0.08)
    plt.close(figure)


def framework_cmap(name, background):
    """Return a colormap whose zero-valued cells blend into the panel."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(background)
    return cmap


def save_framework_localization_view(path, record, gt, c, after_offset, args):
    """Save the minimal square localization panel used in framework diagrams.

    Unlike the analysis layout this deliberately omits axes, title, legend and
    arrows. The two panels therefore read as two compact processing stages:
    quantized-grid localization before offset and sub-pixel localization after
    offset. The positions still use the same evaluator-compatible coordinates.
    """
    background = '#303034'
    side = record['filtered_matrix'].shape[0]
    center = (c - 1) // 2
    detections = (record['refined_detections'] if after_offset
                  else record['raw_detections'])

    # The reconstruction itself is unchanged by offset; only the display
    # palette changes between the two compact stages to make the architecture
    # flow visually clear. Both use the same thresholded CSO candidate grid.
    display = np.ma.masked_where(record['filtered_matrix'] <= 0,
                                 record['filtered_matrix'])
    positive = record['filtered_matrix'][record['filtered_matrix'] > 0]
    vmax = float(np.percentile(positive, 99.2)) if positive.size else 1.0
    vmax = max(vmax, 1.0)
    cmap_name = 'magma' if after_offset else 'gray'

    figure = plt.figure(figsize=(4, 4), dpi=args.dpi, facecolor=background)
    axes = figure.add_axes((0, 0, 1, 1), facecolor=background)
    axes.imshow(display, cmap=framework_cmap(cmap_name, background),
                vmin=0, vmax=vmax, interpolation='nearest', origin='upper')

    # Convert GT LR coordinates into the HR grid. For predictions the raw
    # detection is already at (grid_row, grid_col); the refined detection has
    # the offset added in that same HR grid before this plotting conversion.
    gt_rows = gt[:, 0] * c + center
    gt_cols = gt[:, 1] * c + center
    pred_rows = [det['row'] * c + center for det in detections]
    pred_cols = [det['col'] * c + center for det in detections]

    # Ground truth: yellow ×. Raw prediction: red; refined prediction: blue.
    axes.scatter(gt_cols, gt_rows, marker='x', color=GT_COLOR, s=140,
                 linewidths=2.5, zorder=6)
    if detections:
        axes.scatter(pred_cols, pred_rows, marker='o', facecolors='none',
                     edgecolors=(REFINED_PRED_COLOR if after_offset
                                 else PRED_COLOR),
                     s=100, linewidths=2.0, zorder=7)
    axes.set_xlim(-0.5, side - 0.5)
    axes.set_ylim(side - 0.5, -0.5)
    axes.set_aspect('equal', adjustable='box')
    axes.set_axis_off()
    figure.savefig(path, dpi=args.dpi, facecolor=background,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)


def save_localization_view(path, record, gt, c, after_offset, args):
    if args.localization_layout == 'analysis':
        save_analysis_localization_view(path, record, gt, c, after_offset, args)
    else:
        save_framework_localization_view(path, record, gt, c, after_offset, args)


def metadata_for_record(record, gt, args, c, threshold, offset_limit,
                        selection_mode):
    return {
        'sample_index': record['sample_index'],
        'file_id': record['file_id'],
        'selection_mode': selection_mode,
        'localization_layout': args.localization_layout,
        'selection_criteria': {
            'overall_mean_improvement_must_be_positive': True,
            'majority_improved_or_clear_mean_improvement': True,
            'min_improved_match_ratio': args.min_improved_ratio,
            'min_refined_matches': args.min_refined_matches,
            'min_refined_precision': args.min_refined_precision,
            'min_focal_raw_distance': args.min_raw_distance,
            'min_focal_improvement': args.min_improvement,
            'min_mean_sample_improvement': args.min_sample_improvement,
            'worsened_matches_allowed': True,
        },
        'cso_metric_settings': {
            'brightness_threshold': threshold,
            'c': c,
            'subpixel_offset_limit': offset_limit,
            'match_distance_threshold': args.match_distance_threshold,
        },
        'coordinate_convention': {
            'point_order': '[row, col]',
            'plotting': 'x=col, y=row',
            'analysis_plot_space': (
                'continuous high-resolution grid coordinates; raw prediction '
                'markers are integer grid indices'),
            'raw_grid_to_lr': '[(grid_row-center)/c, (grid_col-center)/c]',
            'refined_grid_to_lr': '[(grid_row+row_offset-center)/c, '
                                   '(grid_col+col_offset-center)/c]',
            'offset_channels': 'channel 0=row, channel 1=col',
            'offset_units': 'high-resolution grid pixels',
            'offset_head': 'tanh(output) * offset_scale inside model; then '
                           'clamped again as CSO_Metrics does',
        },
        'marker_meaning': {
            'yellow_x': 'ground truth',
            'red_hollow_circle': 'original grid prediction',
            'blue_hollow_circle': 'offset-refined prediction',
            'blue_arrow_in_after_offset': 'raw-to-refined offset displacement',
        },
        'ground_truth': [{'row': float(point[0]), 'col': float(point[1]),
                          'brightness': float(point[2])} for point in gt],
        'raw_summary': record['raw_summary'],
        'refined_summary': record['refined_summary'],
        'mean_matched_improvement': record['mean_matched_improvement'],
        'improved_match_count': record['improved_match_count'],
        'worsened_match_count': record['worsened_match_count'],
        'improved_match_ratio': record['improved_match_ratio'],
        'worsened_match_ratio': record['worsened_match_ratio'],
        'focal_pair': record['focal_pair'],
        'raw_matches': serializable_matches(
            record['raw_matches'], gt, record['raw_detections']),
        'refined_matches': serializable_matches(
            record['refined_matches'], gt, record['refined_detections']),
        'matched_pair_comparisons': record['comparisons'],
    }


def json_safe(value):
    """Replace non-finite diagnostic values with JSON null."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()
    if args.batch_size < 1 or args.dpi < 1 or args.match_distance_threshold <= 0:
        raise ValueError('--batch-size, --dpi, and --match-distance-threshold must be positive.')
    if args.min_refined_matches < 1:
        raise ValueError('--min-refined-matches must be at least 1.')
    if not 0 <= args.min_improved_ratio <= 1:
        raise ValueError('--min-improved-ratio must be in [0, 1].')
    if not 0 <= args.min_refined_precision <= 1:
        raise ValueError('--min-refined-precision must be in [0, 1].')
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError('Both --config and --checkpoint must exist.')

    cfg = Config.fromfile(str(config_path))
    dataset_cfg = (cfg.val_dataloader.dataset if args.split == 'val'
                   else cfg.test_dataloader.dataset)
    dataset = DATASETS.build(dataset_cfg)
    metric_cfg = get_metric_cfg(cfg)
    c = int(metric_cfg.get('c', cfg.model.get('c', 3)))
    threshold = (float(args.brightness_threshold) if args.brightness_threshold
                 is not None else float(metric_cfg.get('brightness_threshold', 50)))
    offset_limit = (float(args.subpixel_offset_limit)
                    if args.subpixel_offset_limit is not None
                    else float(metric_cfg.get('subpixel_offset_limit', 0.5)))

    device = resolve_device(args.device)
    model = prepare_model(cfg, checkpoint_path, device)
    record, selection_mode = find_best_sample(
        args, model, dataset, c, threshold, offset_limit)
    selected_index = int(record['sample_index'])
    selected_file_id = str(record['file_id'])

    # Infer the selected sample again to ensure all exported panels use the
    # same sample and retain the evaluator-compatible candidate logic.
    sample = dataset[selected_index]
    with torch.no_grad():
        output = model(mode='predict', **make_batch([sample]))[0]
    matrices, offsets = extract_prediction_maps(output, c)
    gt = as_gt_array(output['targets_GT'][0])
    record = sample_record(matrices[0], offsets[0], gt, c, threshold,
                           offset_limit, args.match_distance_threshold)
    record['sample_index'] = selected_index
    record['file_id'] = selected_file_id
    raw_input = sample['gt_img_11'].detach().float().cpu().numpy().reshape(11, 11)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_original_input(out_dir / '01_original_input.png', raw_input, args.dpi)
    save_intensity_map(out_dir / '02_high_resolution_intensity.png',
                       matrices[0], args.dpi)
    save_localization_view(out_dir / '03_before_offset.png', record, gt, c,
                           after_offset=False, args=args)
    save_localization_view(out_dir / '04_after_offset.png', record, gt, c,
                           after_offset=True, args=args)
    if args.localization_layout == 'analysis':
        save_offset_local_comparison(
            out_dir / '05_before_after_local_zoom.png', record, gt, c, args)
    save_intro_motivation_figure(
        out_dir / '06_intro_motivation.png', raw_input, record, gt, c, args)
    metadata = metadata_for_record(record, gt, args, c, threshold, offset_limit,
                                   selection_mode)
    metadata['introduction_figure'] = {
        'png': '06_intro_motivation.png',
        'pdf': '06_intro_motivation.pdf',
        'panels': [
            'mixed infrared observation',
            'grid-based target recovery with local crop',
            'discrete grid prediction and continuous target location',
        ],
    }
    (out_dir / 'selection.json').write_text(
        json.dumps(json_safe(metadata), indent=2, ensure_ascii=False,
                   allow_nan=False) + '\n',
        encoding='utf-8')

    refined = record['refined_summary']
    focal = record['focal_pair']
    print(f'Selected file_id={selected_file_id}, sample_index={selected_index}, '
          f'mode={selection_mode}')
    print(f'refined recall={refined["recall"]:.3f}, precision='
          f'{refined["precision"]:.3f}, max_distance='
          f'{refined["max_matched_distance"]:.6f}')
    print(f'focal improvement={focal["improvement"]:.6f}, mean matched '
          f'improvement={record["mean_matched_improvement"]:.6f}, improved '
          f'ratio={record["improved_match_ratio"]:.3f}, worsened ratio='
          f'{record["worsened_match_ratio"]:.3f}')
    png_count = 6 if args.localization_layout == 'analysis' else 5
    print(f'Saved {png_count} PNG files, the introduction PDF, and '
          f'selection.json to {out_dir}')


if __name__ == '__main__':
    main()
