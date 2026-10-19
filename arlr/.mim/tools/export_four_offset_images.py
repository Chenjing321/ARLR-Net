"""Export four compact, framework-ready views of one Offset sample.

The output images deliberately contain no title, axes, or legend:

01_original_input.png          low-resolution 11x11 input, smoothly enlarged
02_high_resolution_target.png  sparse 33x33 high-resolution GT intensity
03_before_offset.png           raw-grid candidates and GT markers
04_after_offset.png            intensity candidates after sub-pixel offset
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the checked ARLR/CSO candidate, coordinate, matching, and selection
# logic instead of duplicating a second evaluator implementation here.
from export_successful_offset_sample import (  # noqa: E402
    as_gt_array,
    extract_prediction_maps,
    find_best_sample,
    get_metric_cfg,
    make_batch,
    prepare_model,
    resolve_device,
    sample_record,
)
from arlr.datasets import *  # noqa: E402,F401,F403


GT_COLOR = '#FFD34E'
RAW_PRED_COLOR = '#FF4D5A'
REFINED_PRED_COLOR = '#35D4E8'
PANEL_BACKGROUND = '#000000'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export four clean original/HR/before/after Offset images.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--sample-index', type=int, default=None,
                        help='Use this zero-based sample index instead of searching.')
    parser.add_argument(
        '--selection-json', default=None,
        help='Use sample_index from an existing four-image selection.json; '
             'the recorded file_id is checked against the selected split.')
    parser.add_argument('--target-count', type=int, choices=(1, 2, 3, 4, 5),
                        default=None,
                        help='For automatic selection, restrict candidates to '
                             'samples with exactly this many GT targets.')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples for automatic selection.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default=None)
    parser.add_argument('--brightness-threshold', type=float, default=None)
    parser.add_argument('--subpixel-offset-limit', type=float, default=None)
    parser.add_argument('--match-distance-threshold', type=float, default=0.10)
    parser.add_argument('--min-raw-distance', type=float, default=0.10)
    parser.add_argument('--min-improvement', type=float, default=0.08)
    parser.add_argument('--min-sample-improvement', type=float, default=0.03)
    parser.add_argument('--min-improved-ratio', type=float, default=0.50)
    parser.add_argument('--min-refined-matches', type=int, default=1)
    parser.add_argument('--min-refined-precision', type=float, default=0.80,
                        help='Use 1.0 to reject samples with false alarms.')
    parser.add_argument('--allow-relaxed-fallback', action='store_true')
    parser.add_argument('--offset-zoom-size', type=float, default=5.0,
                        help='Side length of the before/after local crop in '
                             'high-resolution grid pixels (default: 5).')
    parser.add_argument('--offset-zoom-row-shift', type=float, default=1.0,
                        help='Positive high-resolution-grid shift moves the '
                             'shared before/after crop downward (default: 1).')
    parser.add_argument(
        '--export-target-zoom', action='store_true',
        help='Also export 02_high_resolution_target_zoom.png, cropped to a '
             'square region containing every nonzero GT target.')
    parser.add_argument(
        '--target-zoom-only', action='store_true',
        help='Export only 02_high_resolution_target_zoom.png without '
             'overwriting the existing four images or selection.json.')
    parser.add_argument(
        '--target-zoom-padding', type=int, default=1,
        help='Non-negative padding around the GT target bounding box in '
             'high-resolution grid pixels (default: 1).')
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def normalize_uint8(matrix):
    image = np.nan_to_num(np.maximum(matrix, 0), nan=0.0)
    peak = float(image.max())
    if peak > 0:
        image /= peak
    return np.uint8(np.round(image * 255))


def save_raster(path, matrix, dpi, resample):
    image = Image.fromarray(normalize_uint8(matrix), mode='L')
    target_side = max(image.width, int(round(4 * dpi)))
    resampling = getattr(Image, 'Resampling', Image)
    image.resize((target_side, target_side), getattr(resampling, resample)).save(
        path, dpi=(dpi, dpi))


def crop_to_nonzero_square(matrix, padding):
    """Return a padded square crop containing every positive GT cell."""
    matrix = np.asarray(matrix)
    positions = np.argwhere(np.isfinite(matrix) & (matrix > 0))
    if not len(positions):
        raise ValueError('Cannot crop the high-resolution target: no GT exists.')

    height, width = matrix.shape
    row_min = int(positions[:, 0].min())
    row_max = int(positions[:, 0].max())
    col_min = int(positions[:, 1].min())
    col_max = int(positions[:, 1].max())
    padded_height = row_max - row_min + 1 + 2 * padding
    padded_width = col_max - col_min + 1 + 2 * padding
    crop_size = min(max(padded_height, padded_width), height, width)

    def centered_bounds(lower, upper, limit):
        center = (lower + upper) / 2.0
        start = int(math.floor(center - (crop_size - 1) / 2.0))
        start = min(max(start, 0), limit - crop_size)
        return start, start + crop_size

    row_start, row_end = centered_bounds(row_min, row_max, height)
    col_start, col_end = centered_bounds(col_min, col_max, width)
    bounds = (row_start, row_end, col_start, col_end)
    return matrix[row_start:row_end, col_start:col_end], bounds


def read_sample_selection(path):
    selection_path = Path(path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f'Selection JSON not found: {selection_path}')
    try:
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Selection JSON is invalid: {selection_path}') from exc
    if not isinstance(selection, dict) or 'sample_index' not in selection:
        raise KeyError(
            f'Selection JSON must contain sample_index: {selection_path}')
    sample_index = selection['sample_index']
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError(
            f'sample_index must be an integer, got {sample_index!r}.')
    expected_file_id = selection.get('file_id')
    if expected_file_id is not None:
        expected_file_id = str(expected_file_id)
    return sample_index, expected_file_id, selection_path


def compact_axis(matrix, c, gt, detections, path, after_offset, dpi,
                 zoom=None, raw_detections=None):
    """Draw a minimal 33x33 panel in high-resolution grid coordinates."""
    side = matrix.shape[0]
    center = (c - 1) // 2
    display = np.ma.masked_where(matrix <= 0, matrix)
    positive = matrix[matrix > 0]
    vmax = float(np.percentile(positive, 99.2)) if positive.size else 1.0
    vmax = max(vmax, 1.0)
    # Offset changes only localization coordinates, not x_final intensity.
    # Therefore the two panels intentionally share the identical intensity map,
    # colormap, and normalization; only prediction marker locations differ.
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(PANEL_BACKGROUND)

    figure = plt.figure(figsize=(4, 4), dpi=dpi, facecolor=PANEL_BACKGROUND)
    axes = figure.add_axes((0, 0, 1, 1), facecolor=PANEL_BACKGROUND)
    axes.imshow(display, cmap=cmap, vmin=0, vmax=vmax, origin='upper',
                interpolation='nearest')

    # Both GT and detections are [row, col]. Plotting always uses x=col,y=row.
    axes.scatter(gt[:, 1] * c + center, gt[:, 0] * c + center,
                 marker='x', color=GT_COLOR, s=140, linewidths=2.5, zorder=5)
    if after_offset and raw_detections is not None:
        # Draw both endpoints of each Offset vector in the same cropped HR
        # coordinate frame: red is the original grid point, blue is refined.
        axes.scatter([det['col'] * c + center for det in raw_detections],
                     [det['row'] * c + center for det in raw_detections],
                     marker='o', facecolors='none', edgecolors=RAW_PRED_COLOR,
                     s=112, linewidths=2.2, zorder=6)
        raw_by_grid = {(det['grid_row'], det['grid_col']): det
                       for det in raw_detections}
        for refined in detections:
            raw = raw_by_grid[(refined['grid_row'], refined['grid_col'])]
            axes.annotate(
                '',
                xy=(refined['col'] * c + center,
                    refined['row'] * c + center),
                xytext=(raw['col'] * c + center, raw['row'] * c + center),
                arrowprops=dict(arrowstyle='->', color=REFINED_PRED_COLOR,
                                lw=1.6, shrinkA=7, shrinkB=6),
                zorder=7)
        if detections:
            axes.scatter([det['col'] * c + center for det in detections],
                         [det['row'] * c + center for det in detections],
                         marker='o', facecolors='none',
                         edgecolors=REFINED_PRED_COLOR, s=88,
                         linewidths=2.0, zorder=8)
    elif detections:
        pred_rows = [det['row'] * c + center for det in detections]
        pred_cols = [det['col'] * c + center for det in detections]
        axes.scatter(pred_cols, pred_rows, marker='o', facecolors='none',
                     edgecolors=(REFINED_PRED_COLOR if after_offset
                                 else RAW_PRED_COLOR),
                     s=100, linewidths=2.0, zorder=6)
    if zoom is None:
        axes.set_xlim(-0.5, side - 0.5)
        axes.set_ylim(side - 0.5, -0.5)
    else:
        axes.set_xlim(zoom[0], zoom[1])
        axes.set_ylim(zoom[3], zoom[2])
    axes.set_aspect('equal', adjustable='box')
    axes.set_axis_off()
    figure.savefig(path, dpi=dpi, facecolor=PANEL_BACKGROUND,
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)


def offset_zoom_bounds(record, gt, c, side, zoom_size, row_shift=0.0):
    """Build one shared HR-grid crop around the strongest offset correction."""
    focal = record['focal_pair']
    if focal is None:
        return None
    center = (c - 1) // 2
    gt_row, gt_col = gt[focal['gt_index'], :2] * c + center
    raw = next(det for det in record['raw_detections']
               if (det['grid_row'], det['grid_col']) ==
               (focal['grid_row'], focal['grid_col']))
    refined = next(det for det in record['refined_detections']
                   if (det['grid_row'], det['grid_col']) ==
                   (focal['grid_row'], focal['grid_col']))
    rows = np.asarray([gt_row, raw['row'] * c + center,
                       refined['row'] * c + center], dtype=np.float64)
    cols = np.asarray([gt_col, raw['col'] * c + center,
                       refined['col'] * c + center], dtype=np.float64)
    size = max(float(zoom_size), float(rows.max() - rows.min() + 2.0),
               float(cols.max() - cols.min() + 2.0))
    size = min(size, float(side))
    # In image coordinates row increases downward. A small positive shift
    # leaves room below the focal correction for a nearby third target.
    row_center = float((rows.min() + rows.max()) / 2.0 + row_shift)
    col_center = float((cols.min() + cols.max()) / 2.0)

    def clip_bounds(center_value):
        lower = center_value - size / 2.0
        upper = center_value + size / 2.0
        if lower < -0.5:
            upper -= lower + 0.5
            lower = -0.5
        if upper > side - 0.5:
            lower -= upper - (side - 0.5)
            upper = side - 0.5
        return lower, upper

    col_min, col_max = clip_bounds(col_center)
    row_min, row_max = clip_bounds(row_center)
    return col_min, col_max, row_min, row_max


class TargetCountSubset:
    """Index-preserving dataset view used only for automatic sample search."""

    def __init__(self, dataset, original_indices):
        self.dataset = dataset
        self.original_indices = list(original_indices)

    def __len__(self):
        return len(self.original_indices)

    def __getitem__(self, index):
        return self.dataset[self.original_indices[index]]


def selected_record(args, model, dataset, c, threshold, offset_limit):
    if args.sample_index is None:
        search_dataset = dataset
        original_indices = None
        if args.target_count is not None:
            original_indices = [
                index for index in range(len(dataset))
                if int(torch.count_nonzero(dataset[index]['gt']).item()) ==
                args.target_count
            ]
            if not original_indices:
                raise RuntimeError(
                    f'No {args.target_count}-target samples exist in the '
                    f'{args.split} split.')
            search_dataset = TargetCountSubset(dataset, original_indices)
        record, mode = find_best_sample(
            args, model, search_dataset, c, threshold, offset_limit)
        selected_index = int(record['sample_index'])
        if original_indices is not None:
            selected_index = original_indices[selected_index]
            mode = f'{mode}_target_count_{args.target_count}'
        return selected_index, str(record['file_id']), mode
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f'--sample-index must be in [0, {len(dataset) - 1}].')
    sample = dataset[args.sample_index]
    if (args.target_count is not None and
            int(torch.count_nonzero(sample['gt']).item()) != args.target_count):
        raise ValueError(
            f'--sample-index {args.sample_index} does not contain exactly '
            f'{args.target_count} targets.')
    return int(args.sample_index), str(sample['file_id']), 'explicit_sample_index'


def main():
    args = parse_args()
    if args.batch_size < 1 or args.dpi < 1 or args.min_refined_matches < 1:
        raise ValueError('--batch-size, --dpi, and --min-refined-matches must be positive.')
    if args.offset_zoom_size <= 0:
        raise ValueError('--offset-zoom-size must be positive.')
    if args.target_zoom_padding < 0:
        raise ValueError('--target-zoom-padding must be non-negative.')
    if args.sample_index is not None and args.selection_json is not None:
        raise ValueError('Use only one of --sample-index and --selection-json.')
    expected_file_id = None
    selection_source = None
    if args.selection_json is not None:
        args.sample_index, expected_file_id, selection_source = (
            read_sample_selection(args.selection_json))
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
    index, file_id, selection_mode = selected_record(
        args, model, dataset, c, threshold, offset_limit)
    if expected_file_id is not None and file_id != expected_file_id:
        raise ValueError(
            'The selection JSON does not match this dataset split: '
            f'sample_index={index} records file_id={expected_file_id}, '
            f'but the dataset returned file_id={file_id}.')
    if selection_source is not None:
        selection_mode = 'existing_selection_json'
    sample = dataset[index]
    with torch.no_grad():
        output = model(mode='predict', **make_batch([sample]))[0]
    matrices, offsets = extract_prediction_maps(output, c)
    matrix = matrices[0]
    gt = as_gt_array(output['targets_GT'][0])
    record = sample_record(matrix, offsets[0], gt, c, threshold, offset_limit,
                           args.match_distance_threshold)

    target = output['img_gt'][0].detach().float().cpu().numpy()
    side = int(round(math.sqrt(target.size)))
    if side != matrix.shape[0] or side * side != target.size:
        raise ValueError(f'img_gt must match the {matrix.shape} HR map, got {target.shape}.')
    target = target.reshape(side, side)
    raw_input = sample['gt_img_11'].detach().float().cpu().numpy().reshape(11, 11)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.target_zoom_only:
        target_zoom, target_zoom_bounds = crop_to_nonzero_square(
            target, args.target_zoom_padding)
        target_zoom_path = out_dir / '02_high_resolution_target_zoom.png'
        save_raster(target_zoom_path, target_zoom, args.dpi, 'NEAREST')
        print(f'Selected file_id={file_id}, sample_index={index}, '
              f'mode={selection_mode}')
        print(f'Saved target zoom to {target_zoom_path}; '
              f'HR-grid bounds={target_zoom_bounds}')
        return

    save_raster(out_dir / '01_original_input.png', raw_input, args.dpi, 'BICUBIC')
    save_raster(out_dir / '02_high_resolution_target.png', target, args.dpi,
                'NEAREST')
    target_zoom_bounds = None
    if args.export_target_zoom:
        target_zoom, target_zoom_bounds = crop_to_nonzero_square(
            target, args.target_zoom_padding)
        save_raster(out_dir / '02_high_resolution_target_zoom.png',
                    target_zoom, args.dpi, 'NEAREST')
    zoom = offset_zoom_bounds(record, gt, c, matrix.shape[0],
                              args.offset_zoom_size,
                              args.offset_zoom_row_shift)
    compact_axis(record['filtered_matrix'], c, gt, record['raw_detections'],
                 out_dir / '03_before_offset.png', after_offset=False,
                 dpi=args.dpi, zoom=zoom)
    compact_axis(record['filtered_matrix'], c, gt, record['refined_detections'],
                 out_dir / '04_after_offset.png', after_offset=True,
                 dpi=args.dpi, zoom=zoom,
                 raw_detections=record['raw_detections'])

    metadata = {
        'sample_index': index,
        'file_id': file_id,
        'selection_mode': selection_mode,
        'selection_source': (
            str(selection_source) if selection_source is not None else None),
        'requested_target_count': args.target_count,
        'actual_target_count': int(len(gt)),
        'files': [
            '01_original_input.png',
            '02_high_resolution_target.png',
        ] + (['02_high_resolution_target_zoom.png']
             if args.export_target_zoom else []) + [
            '03_before_offset.png',
            '04_after_offset.png',
        ],
        'marker_meaning': {
            'yellow_x': 'ground truth',
            'cyan_hollow_circle_before': 'raw-grid prediction',
            'red_hollow_circle_in_after': 'raw-grid prediction',
            'blue_hollow_circle_in_after': 'offset-refined prediction',
            'blue_arrow_in_after': 'raw-to-refined Offset displacement',
        },
        'coordinate_convention': 'points are [row, col]; plots use x=col, y=row',
        'cso_settings': {
            'brightness_threshold': threshold,
            'c': c,
            'subpixel_offset_limit': offset_limit,
        },
        'before_after_local_zoom_hr_grid': (
            [float(value) for value in zoom] if zoom is not None else None),
        'requested_offset_zoom_size_hr_grid': args.offset_zoom_size,
        'requested_offset_zoom_row_shift_hr_grid': args.offset_zoom_row_shift,
        'target_zoom_bounds_hr_grid': (
            [int(value) for value in target_zoom_bounds]
            if target_zoom_bounds is not None else None),
        'target_zoom_padding_hr_grid': args.target_zoom_padding,
    }
    (out_dir / 'selection.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Selected file_id={file_id}, sample_index={index}, mode={selection_mode}')
    print(f'Saved four images to {out_dir}')


if __name__ == '__main__':
    main()
