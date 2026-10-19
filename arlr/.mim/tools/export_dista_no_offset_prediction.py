"""Export one DISTA-Net no-offset localization view.

The selected sample and local high-resolution-grid crop come from the
``selection.json`` written by ``export_four_offset_images.py``.  The exported
panel therefore aligns directly with that sample's ARLR-Net before/after
Offset views.  It contains only the DISTA-Net final intensity map, GT crosses,
and raw grid prediction circles; DISTA-Net has no sub-pixel offset refinement.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from export_four_offset_images import compact_axis  # noqa: E402
from export_successful_offset_sample import (  # noqa: E402
    as_gt_array,
    detections_from_map,
    get_metric_cfg,
    make_batch,
    prepare_model,
    resolve_device,
)
from arlr.datasets import *  # noqa: E402,F401,F403


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export DISTA-Net raw-grid predictions in the same local '
                    'crop as a ARLR-Net Offset sample.')
    parser.add_argument('--config', required=True,
                        help='DISTA-Net config file.')
    parser.add_argument('--checkpoint', required=True,
                        help='DISTA-Net checkpoint file.')
    parser.add_argument('--selection-json', required=True,
                        help='selection.json from export_four_offset_images.py.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--out-name', default='06_dista_net_no_offset_prediction.png',
        help='Plain PNG filename inside --out-dir.')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--brightness-threshold', type=float, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dpi', type=int, default=300)
    return parser.parse_args()


def load_selection(path):
    selection_path = Path(path).expanduser().resolve()
    if not selection_path.is_file():
        raise FileNotFoundError(f'Selection JSON not found: {selection_path}')
    try:
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f'Selection JSON is invalid: {selection_path}') from exc
    if not isinstance(selection, dict):
        raise TypeError('Selection JSON must contain an object.')

    sample_index = selection.get('sample_index')
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError(
            f'sample_index must be an integer, got {sample_index!r}.')
    file_id = selection.get('file_id')
    if file_id is None:
        raise KeyError('Selection JSON must contain file_id.')

    zoom = selection.get('before_after_local_zoom_hr_grid')
    if not isinstance(zoom, list) or len(zoom) != 4:
        raise ValueError(
            'Selection JSON must contain the four-value '
            'before_after_local_zoom_hr_grid field.')
    zoom = tuple(float(value) for value in zoom)
    if not all(math.isfinite(value) for value in zoom):
        raise ValueError('Local zoom bounds must all be finite.')
    return sample_index, str(file_id), zoom, selection_path


def output_name(name):
    path = Path(name)
    if path.name != name or path.suffix.lower() != '.png':
        raise ValueError('--out-name must be a plain filename ending in .png.')
    return name


def prediction_matrix(output, c):
    if 'x_final' not in output or not torch.is_tensor(output['x_final']):
        raise KeyError('DISTA-Net predict output must contain tensor x_final.')
    prediction = output['x_final'].detach().float().cpu()
    if prediction.ndim != 2 or prediction.shape[0] != 1:
        raise ValueError(
            f'x_final must have shape [1, N], got {tuple(prediction.shape)}.')
    side = int(round(math.sqrt(prediction.shape[1])))
    if side * side != prediction.shape[1] or side != 11 * c:
        raise ValueError(
            f'x_final must describe one {(11 * c)}x{(11 * c)} map for c={c}, '
            f'got {tuple(prediction.shape)}.')
    return prediction[0].numpy().reshape(side, side)


def main():
    args = parse_args()
    if args.dpi < 1:
        raise ValueError('--dpi must be positive.')

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(f'Checkpoint is empty: {checkpoint_path}')

    sample_index, expected_file_id, zoom, selection_path = load_selection(
        args.selection_json)
    cfg = Config.fromfile(str(config_path))
    dataset_cfg = (cfg.val_dataloader.dataset if args.split == 'val'
                   else cfg.test_dataloader.dataset)
    dataset = DATASETS.build(dataset_cfg)
    if not 0 <= sample_index < len(dataset):
        raise IndexError(
            f'sample_index must be in [0, {len(dataset) - 1}], '
            f'got {sample_index}.')
    sample = dataset[sample_index]
    actual_file_id = str(sample.get('file_id', ''))
    if actual_file_id != expected_file_id:
        raise ValueError(
            'The selection JSON does not match this dataset split: '
            f'sample_index={sample_index} records file_id={expected_file_id}, '
            f'but the dataset returned file_id={actual_file_id}.')

    metric_cfg = get_metric_cfg(cfg)
    c = int(metric_cfg.get('c', cfg.model.get('c', 3)))
    threshold = (float(args.brightness_threshold)
                 if args.brightness_threshold is not None
                 else float(metric_cfg.get('brightness_threshold', 50)))
    offset_limit = float(metric_cfg.get('subpixel_offset_limit', 0.5))

    device = resolve_device(args.device)
    model = prepare_model(cfg, checkpoint_path, device)
    with torch.no_grad():
        output = model(mode='predict', **make_batch([sample]))[0]
    matrix = prediction_matrix(output, c)
    gt = as_gt_array(output['targets_GT'][0])
    detections, filtered = detections_from_map(
        matrix, c, threshold, offset_limit, subpixel_offset=None)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / output_name(args.out_name)
    compact_axis(filtered, c, gt, detections, output_path,
                 after_offset=False, dpi=args.dpi, zoom=zoom)
    print(f'Saved {output_path}')
    print(f'sample_index={sample_index}; file_id={actual_file_id}; '
          f'candidate_count={len(detections)}; '
          f'brightness_threshold={threshold}; c={c}; device={device}; '
          f'selection_json={selection_path}; zoom={zoom}')


if __name__ == '__main__':
    main()
