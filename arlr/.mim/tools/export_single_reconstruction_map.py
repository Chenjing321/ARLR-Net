"""Export one clean high-resolution reconstruction-intensity map.

The script deliberately exports only the model reconstruction: it does not
draw ground truth, candidates, localization offsets, labels, axes, or a color
bar.  Its dataset construction and ``mode='predict'`` invocation match the
project's existing visualization scripts.
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
import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.runner.checkpoint import load_checkpoint
from PIL import Image


# Allow ``python tools/export_single_reconstruction_map.py ...`` from any
# working directory while retaining the project's normal module imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arlr.datasets import *  # noqa: E402,F401,F403
from arlr.evaluation import *  # noqa: E402,F401,F403
from arlr.models import *  # noqa: E402,F401,F403


FINAL_OUTPUT_CANDIDATES = ('x_final', 'pred', 'prediction', 'x_output')
EXPORT_SIZE_INCHES = 4.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export one annotation-free reconstruction intensity map.')
    parser.add_argument('--config', required=True, help='Model config file.')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint.')
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument('--sample-idx', type=int,
                              help='Zero-based index in the test dataset.')
    sample_group.add_argument(
        '--selection-json',
        help='Read sample_index from export_four_offset_images.py output. '
             'The recorded file_id is checked against the test dataset.')
    parser.add_argument('--out-dir', required=True,
                        help='Directory for the exported image.')
    parser.add_argument(
        '--out-name', default=None,
        help='PNG filename inside --out-dir (default: '
             'sample_{idx}_reconstruction_map.png).')
    parser.add_argument('--cmap', default='gray',
                        help='Matplotlib colormap (default: gray).')
    parser.add_argument('--map-type', choices=('raw', 'filtered'),
                        default='raw',
                        help='raw: x_final; filtered: values below the '
                             'test evaluator brightness threshold are zeroed.')
    parser.add_argument(
        '--crop-to-targets', action='store_true',
        help='Crop to one square region containing every GT target before '
             'upscaling. GT is used only to choose the crop and is not drawn.')
    parser.add_argument(
        '--target-crop-padding', type=int, default=1,
        help='Non-negative padding around the target bounding box in HR-grid '
             'pixels (default: 1).')
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument('--normalize', dest='normalize',
                                 action='store_true',
                                 help='Scale non-negative intensities to [0, 1] '
                                      '(default).')
    normalize_group.add_argument('--no-normalize', dest='normalize',
                                 action='store_false',
                                 help='Keep the displayed non-negative intensity '
                                      'scale instead of normalizing it.')
    parser.set_defaults(normalize=True)
    parser.add_argument('--dpi', type=int, default=300,
                        help='PNG/PDF resolution (default: 300).')
    parser.add_argument('--save-pdf', action='store_true',
                        help='Also save sample_{idx}_reconstruction_map.pdf.')
    parser.add_argument('--device', default=None,
                        help='Torch device, e.g. cuda or cpu (default: cuda if '
                             'available, otherwise cpu).')
    return parser.parse_args()


def resolve_sample_selection(args):
    """Resolve one test index, optionally from the four-image metadata."""
    if args.selection_json is None:
        return int(args.sample_idx), None, None

    selection_path = Path(args.selection_json).expanduser().resolve()
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
    sample_idx = selection['sample_index']
    if isinstance(sample_idx, bool) or not isinstance(sample_idx, int):
        raise TypeError(
            f'sample_index must be an integer, got {sample_idx!r}.')
    expected_file_id = selection.get('file_id')
    if expected_file_id is not None:
        expected_file_id = str(expected_file_id)
    return sample_idx, expected_file_id, selection_path


def resolve_png_name(requested_name, sample_idx):
    if requested_name is None:
        return f'sample_{sample_idx}_reconstruction_map.png'
    output_name = Path(requested_name)
    if output_name.name != requested_name or output_name.suffix.lower() != '.png':
        raise ValueError('--out-name must be a plain filename ending in .png.')
    return requested_name


def get_first_cso_evaluator_cfg(cfg):
    """Return the CSO evaluator config used by the test pipeline, if any."""
    evaluator = cfg.get('test_evaluator', cfg.get('val_evaluator', {}))
    if isinstance(evaluator, (list, tuple)):
        for metric_cfg in evaluator:
            if metric_cfg.get('type') == 'CSO_Metrics':
                return metric_cfg
        return evaluator[0] if evaluator else {}
    return evaluator or {}


def make_batch(sample):
    """Use the same list-based batch format as the existing visualizers."""
    return {key: [value] for key, value in sample.items()}


def unwrap_predict_output(raw_output):
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            raise RuntimeError('The model returned an empty prediction list.')
        return raw_output[0]
    return raw_output


def extract_final_prediction(output):
    """Find the final reconstruction tensor, preferring the project x_final."""
    if torch.is_tensor(output):
        return output, 'tensor_output'
    if not isinstance(output, dict):
        raise TypeError('Unsupported prediction output type: '
                        f'{type(output).__name__}.')
    for key in FINAL_OUTPUT_CANDIDATES:
        value = output.get(key)
        if torch.is_tensor(value):
            return value, key
    candidates = [
        (key, value) for key, value in output.items()
        if torch.is_tensor(value) and value.dim() >= 2
    ]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError('Could not find the final reconstruction tensor. Expected '
                   f'one of {FINAL_OUTPUT_CANDIDATES}; got keys '
                   f'{list(output.keys())}.')


def prediction_to_map(prediction):
    """Convert a one-sample final prediction to its square HR map."""
    prediction = prediction.detach().float().cpu()
    if prediction.dim() == 1:
        prediction = prediction.unsqueeze(0)
    if prediction.dim() > 2:
        prediction = prediction.reshape(prediction.shape[0], -1)
    if prediction.dim() != 2 or prediction.shape[0] < 1:
        raise ValueError('Final prediction must flatten to [B, N], got '
                         f'{tuple(prediction.shape)}.')
    side = int(round(math.sqrt(prediction.shape[1])))
    if side * side != prediction.shape[1]:
        raise ValueError('Final prediction length is not square: '
                         f'{prediction.shape[1]}.')
    return prediction[0].numpy().reshape(side, side)


def filtered_map(matrix, cfg):
    """Match the project visualizer's filtered reconstruction-map behavior."""
    metric_cfg = get_first_cso_evaluator_cfg(cfg)
    threshold = float(metric_cfg.get('brightness_threshold', 50))
    result = np.asarray(matrix, dtype=np.float32).copy()
    result[result < threshold] = 0.0
    return result, threshold


def prepare_display_map(matrix, normalize):
    """Make a black-background intensity map without adding any annotations."""
    display = np.asarray(matrix, dtype=np.float32).copy()
    # Reconstruction intensity is non-negative; clipping makes zero-valued
    # background exactly black and avoids negative reconstruction artifacts
    # being shown as a bright/colored background.
    display[~np.isfinite(display)] = 0.0
    display = np.maximum(display, 0.0)
    if normalize:
        peak = float(display.max())
        if peak > 0:
            display /= peak
    return display


def crop_display_to_targets(display, target, padding):
    """Crop a display map to a padded square around all nonzero GT cells."""
    target = torch.as_tensor(target).detach().float().cpu()
    if target.numel() != display.size:
        raise ValueError(
            f'GT target has {target.numel()} values, but the reconstruction '
            f'map has {display.size}.')
    target = target.reshape(display.shape).numpy()
    positions = np.argwhere(np.isfinite(target) & (target > 0))
    if not len(positions):
        raise ValueError('Cannot crop to targets because this sample has no GT.')

    height, width = display.shape
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
    return display[row_start:row_end, col_start:col_end], bounds


def colorize(display, cmap_name, normalize):
    try:
        # ``plt.get_cmap`` also supports the older Matplotlib versions used
        # by some OpenMMLab environments.
        cmap = plt.get_cmap(cmap_name)
    except ValueError as exc:
        raise ValueError(f'Unknown matplotlib colormap: {cmap_name}') from exc

    vmax = 1.0 if normalize else max(float(display.max()), 1.0)
    rgba = cmap(np.clip(display / vmax, 0.0, 1.0), bytes=True)
    # Guarantee a black zero background even for a non-gray user colormap.
    rgba[display <= 0] = (0, 0, 0, 255)
    return rgba


def render_export_rgba(rgba, dpi):
    """Upscale the small HR grid for a paper-ready raster without margins."""
    image = Image.fromarray(rgba, mode='RGBA')
    height, width = rgba.shape[:2]
    longest_side = max(height, width)
    target_side = max(longest_side, int(round(EXPORT_SIZE_INCHES * dpi)))
    if target_side == longest_side:
        return image
    scale = target_side / longest_side
    target_size = (int(round(width * scale)), int(round(height * scale)))
    resampling = getattr(Image, 'Resampling', Image)
    return image.resize(target_size, resample=resampling.NEAREST)


def save_png(image, output_path, dpi):
    image.save(output_path, dpi=(dpi, dpi))


def save_pdf(image, output_path, dpi):
    """Save a borderless raster image in a PDF page of the map's aspect."""
    width, height = image.size
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi,
                         frameon=False, facecolor='black')
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(image, interpolation='nearest', origin='upper')
    axes.set_axis_off()
    figure.savefig(output_path, dpi=dpi, facecolor='black',
                   bbox_inches=None, pad_inches=0)
    plt.close(figure)


def resolve_device(requested_device):
    if requested_device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(requested_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device requests CUDA, but CUDA is unavailable.')
    return device


def prepare_model(cfg, checkpoint, device):
    # ARLR stores Phi/Qinit as plain tensor attributes and uses a module-level
    # device in forward(), so mirror the existing comparison visualizer setup.
    if cfg.model.get('type') == 'ARLR':
        arlr_module = importlib.import_module('arlr.models.editor.arlr')
        arlr_module.device = device

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location=device)
    model.to(device)
    for attr_name in ('Phi', 'Qinit'):
        value = getattr(model, attr_name, None)
        if torch.is_tensor(value):
            setattr(model, attr_name, value.to(device))
    model.eval()
    return model


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    sample_idx, expected_file_id, selection_path = resolve_sample_selection(args)
    if not config_path.is_file():
        raise FileNotFoundError(f'Config not found: {config_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    if checkpoint_path.stat().st_size == 0:
        raise ValueError(f'Checkpoint is empty: {checkpoint_path}')
    if args.dpi <= 0:
        raise ValueError(f'--dpi must be positive, got {args.dpi}.')
    if args.target_crop_padding < 0:
        raise ValueError('--target-crop-padding must be non-negative.')

    cfg = Config.fromfile(str(config_path))
    if 'test_dataloader' not in cfg or 'dataset' not in cfg.test_dataloader:
        raise KeyError('The config must define test_dataloader.dataset.')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    if not 0 <= sample_idx < len(dataset):
        raise IndexError(f'--sample-idx must be in [0, {len(dataset) - 1}], '
                         f'got {sample_idx}.')

    device = resolve_device(args.device)
    model = prepare_model(cfg, checkpoint_path, device)
    sample = dataset[sample_idx]
    actual_file_id = str(sample.get('file_id', ''))
    if expected_file_id is not None and actual_file_id != expected_file_id:
        raise ValueError(
            'The selection JSON does not match this test dataset: '
            f'sample_index={sample_idx} records file_id={expected_file_id}, '
            f'but the dataset returned file_id={actual_file_id}.')
    with torch.no_grad():
        output = unwrap_predict_output(model(mode='predict', **make_batch(sample)))
    prediction, prediction_field = extract_final_prediction(output)
    matrix = prediction_to_map(prediction)
    threshold = None
    if args.map_type == 'filtered':
        matrix, threshold = filtered_map(matrix, cfg)
    display = prepare_display_map(matrix, args.normalize)
    crop_bounds = None
    if args.crop_to_targets:
        display, crop_bounds = crop_display_to_targets(
            display, sample['gt'], args.target_crop_padding)
    rgba = colorize(display, args.cmap, args.normalize)
    export_image = render_export_rgba(rgba, args.dpi)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_name = resolve_png_name(args.out_name, sample_idx)
    png_path = out_dir / png_name
    save_png(export_image, png_path, args.dpi)
    print(f'Saved {png_path}')
    if args.save_pdf:
        pdf_path = png_path.with_suffix('.pdf')
        save_pdf(export_image, pdf_path, args.dpi)
        print(f'Saved {pdf_path}')
    print(f'sample_index={sample_idx}; file_id={actual_file_id}; '
          f'selection_json={selection_path}')
    print(f'final_prediction_field={prediction_field}; '
          f'map_type={args.map_type}; '
          f'filter_threshold={threshold}; '
          f'normalize={args.normalize}; '
          f'target_crop_bounds={crop_bounds}; '
          f'device={device}')


if __name__ == '__main__':
    main()
