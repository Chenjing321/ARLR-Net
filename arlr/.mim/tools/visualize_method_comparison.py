import argparse
import csv
import importlib
import json
import math
import sys
import time
import traceback
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS
from mmengine.runner.checkpoint import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arlr.datasets import *  # noqa: F401,F403,E402
from arlr.evaluation import *  # noqa: F401,F403,E402
from arlr.models import *  # noqa: F401,F403,E402
from tools.utils import read_targets_from_xml  # noqa: E402


METHOD_SPECS = OrderedDict([
    ('ISTA', dict(config='configs/c_3/cso_ablation_07_full.py')),
    ('CTNet', dict(
        config='configs/c_3/ctnet.py',
        checkpoint='work_dir/pth/cso_compare_c3/ctnet/'
                   'best_cso_metric_mAP_epoch_64.pth')),
    ('SRFBN', dict(
        config='configs/c_3/srfbn.py',
        checkpoint='work_dir/pth/cso_compare_c3/srfbn/'
                   'best_cso_metric_mAP_epoch_98.pth')),
    ('SAN', dict(
        config='configs/c_3/san.py',
        checkpoint='work_dir/pth/cso_compare_c3/san/'
                   'best_cso_metric_mAP_epoch_10.pth')),
    ('FeNet', dict(
        config='configs/c_3/fenet.py',
        checkpoint='work_dir/pth/cso_compare_c3/fenet/'
                   'best_cso_metric_mAP_epoch_65.pth')),
    ('SDSNet', dict(
        config='configs/c_3/sdsnet.py',
        checkpoint='work_dir/pth/cso_compare_c3/sdsnet/'
                   'best_cso_metric_mAP_epoch_20.pth')),
    ('GGLNet', dict(
        config='configs/c_3/gglnet.py',
        checkpoint='work_dir/pth/cso_compare_c3/gglnet/'
                   'best_cso_metric_mAP_epoch_25.pth')),
    ('SD2Net', dict(
        config='configs/c_3/sd2net.py',
        checkpoint='work_dir/pth/cso_compare_c3/sd2net/'
                   'best_cso_metric_mAP_epoch_44.pth')),
    ('ISTA-Net+', dict(
        config='configs/c_3/istanetplus.py',
        checkpoint='work_dir/pth/cso_compare_c3/istanetplus/'
                   'best_cso_metric_mAP_epoch_45.pth')),
    ('FISTA-Net', dict(
        config='configs/c_3/fista.py',
        checkpoint='work_dir/pth/cso_compare_c3/fista/'
                   'best_cso_metric_mAP_epoch_100.pth')),
    ('LISTA', dict(
        config='configs/c_3/lista.py',
        checkpoint='work_dir/pth/cso_compare_c3/lista/'
                   'best_cso_metric_mAP_epoch_99.pth')),
    ('DISTA-Net', dict(
        config='configs/c_3/cso_reference_original_impl.py',
        checkpoint='work_dir/pth/cso_ablation/original_impl/'
                   'best_cso_metric_mAP_epoch_62.pth')),
    ('ARLR-Net', dict(
        config='configs/c_3/cso_ablation_07_full.py',
        checkpoint='work_dir/pth/cso_ablation/07_full/'
                   'best_cso_metric_mAP_epoch_48.pth')),
])

RECON_METHOD_ORDER = [
    'GT',
    'ARLR-Net',
    'DISTA-Net',
    'CTNet',
    'SRFBN',
    'SAN',
    'ISTA-Net+',
    'FISTA-Net',
]
FINAL_DETECTION_METHOD_ORDER = [
    'GT',
    'ARLR-Net',
    'DISTA-Net',
    'CTNet',
    'SRFBN',
    'SAN',
    'ISTA-Net+',
    'FISTA-Net',
]
RECON_TARGET_ROWS = OrderedDict([
    (3, 'Three Targets'),
    (4, 'Four Targets'),
    (5, 'Five Targets'),
])
RECON_DISTINCT_ERR_GAIN_THRESHOLDS = {
    3: 0.030,
    4: 0.025,
    5: 0.020,
}
RECON_DISTINCT_FILTERED_DIFF_THRESHOLDS = {
    3: 0.055,
    4: 0.045,
    5: 0.035,
}

HIGHLIGHT_COLORS = {
    'ARLR-Net': '#E84A5F',
    'DISTA-Net': '#2A6FBB',
}
DEFAULT_COLOR = '#606C76'
GT_COLOR = '#F6C343'
PRED_COLOR = '#47C5FF'
ERROR_LINE_COLOR = '#FFFFFF'
PANEL_FACE_COLOR = '#05070A'
PLOT_DOMAIN = (-0.5, 10.5)
FINAL_OUTPUT_CANDIDATES = (
    'x_final',
    'pred',
    'prediction',
    'x_output',
    'output',
    'out',
    'result',
    'sr',
    'img_pred',
    'reconstruction',
)


class MethodSkip(RuntimeError):
    pass


def configure_matplotlib_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize true c=3 CSO localization predictions from '
                    'ARLR-Net and comparison methods.')
    parser.add_argument(
        '--out-dir',
        default='work_dir/visual_compare',
        help='Directory for generated figures and summaries.')
    parser.add_argument(
        '--dataset-config',
        default='configs/c_3/cso_ablation_07_full.py',
        help='Config used to build the c=3 test dataset.')
    parser.add_argument(
        '--methods',
        default=','.join(METHOD_SPECS.keys()),
        help='Comma-separated method names to run. Unknown names are ignored.')
    parser.add_argument(
        '--recon-methods',
        default=','.join(RECON_METHOD_ORDER),
        help='Comma-separated method names to show in '
             'reconstruction_comparison_grid, in display order. LISTA is '
             'ignored for the reconstruction grid by default.')
    parser.add_argument(
        '--max-scan',
        type=int,
        default=512,
        help='Maximum number of test samples to infer for automatic case '
             'selection. Use -1 for the full test dataset.')
    parser.add_argument(
        '--sample-indices',
        default=None,
        help='Optional comma-separated dataset indices. When set, only these '
             'samples are inferred and selected from.')
    parser.add_argument('--num-cases', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--brightness-threshold', type=float, default=None)
    parser.add_argument('--distance-thr', type=float, default=0.25)
    parser.add_argument('--max-crowd-distance', type=float, default=2.0)
    parser.add_argument('--plot-topk-dets', type=int, default=80)
    parser.set_defaults(plot_matched_only=True)
    parser.add_argument(
        '--plot-matched-only',
        dest='plot_matched_only',
        action='store_true',
        help='For the ARLR/DISTA zoom figure, draw only predictions matched '
             'to GT. Enabled by default.')
    parser.add_argument(
        '--plot-all-predictions',
        dest='plot_matched_only',
        action='store_false',
        help='Disable matched-only display in the ARLR/DISTA zoom figure.')
    parser.add_argument(
        '--hide-intensity-background',
        action='store_true',
        help='Hide final intensity-map background in the ARLR/DISTA zoom '
             'figure and use a clean dark background.')
    parser.add_argument(
        '--background-cmap',
        choices=('gray', 'magma', 'inferno', 'plasma'),
        default='magma',
        help='Colormap for final intensity-map backgrounds in zoom figures.')
    parser.add_argument(
        '--background-alpha',
        type=float,
        default=1.0,
        help='Alpha for localization intensity-map backgrounds. Use 1.0 for '
             'paper figures with a clean black panel background.')
    parser.add_argument(
        '--clean-localization-style',
        action='store_true',
        help='Use a clean paper style for localization figures: black panel '
             'backgrounds, opaque intensity maps, darker low values, clearer '
             'GT/prediction markers, and no error lines.')
    parser.set_defaults(save_reconstruction_aligned_localization=True)
    parser.add_argument(
        '--save-reconstruction-aligned-localization',
        dest='save_reconstruction_aligned_localization',
        action='store_true',
        help='Save localization comparison using exactly the samples selected '
             'for reconstruction_comparison_grid. Enabled by default.')
    parser.add_argument(
        '--no-save-reconstruction-aligned-localization',
        dest='save_reconstruction_aligned_localization',
        action='store_false',
        help='Do not save the reconstruction-aligned localization comparison.')
    parser.add_argument(
        '--recon-map-type',
        choices=('raw', 'filtered'),
        default='filtered',
        help='Map shown in reconstruction_comparison_grid. "filtered" uses '
             'the final intensity map after the brightness_threshold rule '
             'used by localization evaluation; "raw" shows the raw final '
             'output matrix.')
    parser.add_argument(
        '--recon-sample-indices-by-count',
        default=None,
        help='Optional manual reconstruction-grid samples in the form '
             '3:idx,4:idx,5:idx. Matching rows are forced to those dataset '
             'indices when their gt_count is valid.')
    parser.set_defaults(save_recon_candidate_preview=True)
    parser.add_argument(
        '--save-recon-candidate-preview',
        dest='save_recon_candidate_preview',
        action='store_true',
        help='Save a GT/ARLR/DISTA preview grid for top Five Targets '
             'reconstruction candidates. Enabled by default.')
    parser.add_argument(
        '--no-save-recon-candidate-preview',
        dest='save_recon_candidate_preview',
        action='store_false',
        help='Do not save the Five Targets reconstruction candidate preview.')
    parser.add_argument(
        '--recon-candidate-preview-top',
        type=int,
        default=12,
        help='Number of Five Targets reconstruction candidates to show in '
             'the preview figure.')
    parser.set_defaults(recon_strict_position=True)
    parser.add_argument(
        '--recon-strict-position',
        dest='recon_strict_position',
        action='store_true',
        help='For reconstruction grid sample selection, first require ARLR '
             'recall=1, filtered det_count gap=0, and small localization '
             'error. Enabled by default.')
    parser.add_argument(
        '--no-recon-strict-position',
        dest='recon_strict_position',
        action='store_false',
        help='Disable the strict ARLR position-first reconstruction sample '
             'selection pools.')
    parser.add_argument(
        '--show-unmatched-faint',
        action='store_true',
        help='When matched-only plotting is enabled, also show unmatched '
             'candidate predictions as small faint points.')
    parser.add_argument(
        '--show-error-lines',
        action='store_true',
        help='Draw line segments between matched GT and prediction points. '
             'Disabled by default for cleaner paper figures.')
    parser.add_argument(
        '--min-recall-gain',
        type=float,
        default=0.0,
        help='Prefer cases where ARLR recall exceeds DISTA recall by at '
             'least this amount.')
    parser.add_argument(
        '--min-error-gain',
        type=float,
        default=0.0,
        help='Prefer cases where ARLR mean localization error is lower than '
             'DISTA by at least this amount.')
    parser.add_argument(
        '--min-dista-recall',
        type=float,
        default=0.25,
        help='Prefer cases where DISTA recall is at least this value.')
    parser.add_argument(
        '--max-dista-recall',
        type=float,
        default=0.9,
        help='Prefer cases where DISTA recall is no higher than this value.')
    parser.add_argument(
        '--min-arlr-recall',
        type=float,
        default=0.75,
        help='Prefer cases where ARLR recall is at least this value.')
    parser.add_argument(
        '--selection-mode',
        choices=(
            'advantage', 'random', 'balanced', 'arlr_success',
            'arlr_distinct', 'recon_arlr_distinct'),
        default='balanced',
        help='Case selection strategy. "balanced" is recommended for paper '
             'figures.')
    parser.add_argument(
        '--max-arlr-err',
        type=float,
        default=0.10,
        help='In arlr_success mode, prefer ARLR-Net cases with mean GT '
             'distance no larger than this value.')
    parser.add_argument(
        '--min-arlr-success-recall',
        type=float,
        default=1.0,
        help='In arlr_success mode, prefer ARLR-Net cases with recall at '
             'least this value.')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Fixed seed used by balanced and random selection modes.')
    parser.set_defaults(random_crowded_only=True)
    parser.add_argument(
        '--random-crowded-only',
        dest='random_crowded_only',
        action='store_true',
        help='In random mode, sample from crowded cases when possible. '
             'Enabled by default.')
    parser.add_argument(
        '--random-all',
        dest='random_crowded_only',
        action='store_false',
        help='In random mode, sample from all cases with both methods instead '
             'of restricting to crowded cases.')
    parser.add_argument('--ista-lambda', type=float, default=1.0)
    parser.add_argument('--ista-max-iter', type=int, default=3000)
    parser.add_argument('--ista-tol', type=float, default=1e-4)
    parser.add_argument(
        '--skip-ista',
        action='store_true',
        help='Skip the traditional ISTA baseline if a faster neural-only '
             'visualization is desired.')
    parser.add_argument(
        '--allow-empty-checkpoints',
        action='store_true',
        help='Try loading checkpoint paths even when their file size is zero.')
    return parser.parse_args()


def warn(message):
    print(f'WARNING: {message}', flush=True)


def info(message):
    print(message, flush=True)


def as_path(path):
    return (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path)


def parse_method_names(value):
    requested = [item.strip() for item in value.split(',') if item.strip()]
    names = []
    known_lower = {name.lower(): name for name in METHOD_SPECS}
    for item in requested:
        key = known_lower.get(item.lower())
        if key is None:
            warn(f'Unknown method "{item}" was requested; skipping it.')
            continue
        if key not in names:
            names.append(key)
    return names


def parse_recon_method_names(value):
    requested = [item.strip() for item in value.split(',') if item.strip()]
    names = []
    known_lower = {name.lower(): name for name in METHOD_SPECS}
    known_lower['gt'] = 'GT'
    for item in requested:
        key = known_lower.get(item.lower())
        if key is None:
            warn(f'Unknown reconstruction method "{item}" was requested; '
                 'skipping it.')
            continue
        if key == 'LISTA':
            warn('LISTA is skipped in reconstruction_comparison_grid because '
                 'its reconstruction map is visually noisy for this figure.')
            continue
        if key not in names:
            names.append(key)
    if 'GT' not in names:
        names.insert(0, 'GT')
    return names


def parse_recon_sample_indices_by_count(value):
    mapping = {}
    if not value:
        return mapping
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            warn(f'Invalid --recon-sample-indices-by-count item "{item}"; '
                 'expected count:index.')
            continue
        count_text, index_text = item.split(':', 1)
        try:
            gt_count = int(count_text.strip())
            sample_idx = int(index_text.strip())
        except ValueError:
            warn(f'Invalid --recon-sample-indices-by-count item "{item}"; '
                 'count and index must be integers.')
            continue
        if gt_count not in RECON_TARGET_ROWS:
            warn(f'Ignoring manual reconstruction sample for gt_count='
                 f'{gt_count}; expected one of {list(RECON_TARGET_ROWS)}.')
            continue
        mapping[gt_count] = sample_idx
    return mapping


def parse_indices(value, dataset_len, max_scan):
    if value:
        indices = []
        for item in value.split(','):
            item = item.strip()
            if not item:
                continue
            idx = int(item)
            if idx < 0 or idx >= dataset_len:
                warn(f'Sample index {idx} is outside [0, {dataset_len}); '
                     'skipping it.')
                continue
            indices.append(idx)
        return indices

    if max_scan is None or max_scan < 0:
        count = dataset_len
    else:
        count = min(max_scan, dataset_len)
    return list(range(count))


def get_first_cso_evaluator_cfg(cfg):
    evaluator = cfg.get('test_evaluator', cfg.get('val_evaluator', {}))
    if isinstance(evaluator, (list, tuple)):
        for metric_cfg in evaluator:
            if metric_cfg.get('type') == 'CSO_Metrics':
                return metric_cfg
        return evaluator[0] if evaluator else {}
    return evaluator or {}


def resolve_metric_cfg(cfg, args):
    metric_cfg = dict(get_first_cso_evaluator_cfg(cfg))
    if 'c' not in metric_cfg:
        model_cfg = cfg.get('model', {})
        metric_cfg['c'] = int(model_cfg.get('c', model_cfg.get('scale', 3)))
    if 'brightness_threshold' not in metric_cfg:
        metric_cfg['brightness_threshold'] = 50
    if args.brightness_threshold is not None:
        metric_cfg['brightness_threshold'] = args.brightness_threshold
    return metric_cfg


def build_dataset(config_path):
    cfg = Config.fromfile(str(config_path))
    if 'test_dataloader' not in cfg or 'dataset' not in cfg.test_dataloader:
        raise KeyError(f'{config_path} does not define test_dataloader.dataset')
    return cfg, DATASETS.build(cfg.test_dataloader.dataset)


def sample_to_numpy(sample, key):
    value = sample[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def make_batch(samples):
    return {
        key: [sample[key] for sample in samples]
        for key in samples[0].keys()
    }


def load_phi(phi_path):
    data = sio.loadmat(str(phi_path))
    if 'phi' not in data:
        keys = sorted(key for key in data if not key.startswith('__'))
        raise KeyError(f'Cannot find "phi" in {phi_path}; keys={keys}')
    return np.asarray(data['phi'], dtype=np.float64)


def summarize_value(value):
    if torch.is_tensor(value):
        return {
            'type': 'Tensor',
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            'type': 'ndarray',
            'shape': list(value.shape),
            'dtype': str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            'type': 'dict',
            'keys': list(value.keys()),
            'fields': {
                key: summarize_value(item)
                for key, item in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        item = summarize_value(value[0]) if value else None
        return {
            'type': type(value).__name__,
            'len': len(value),
            'first': item,
        }
    return {'type': type(value).__name__}


def unwrap_predict_output(raw_output, method):
    if isinstance(raw_output, (list, tuple)):
        if not raw_output:
            raise KeyError(f'{method}: predict output is an empty {type(raw_output).__name__}.')
        return raw_output[0]
    return raw_output


def extract_final_prediction_tensor(output, method):
    if torch.is_tensor(output):
        return output, 'tensor_output'
    if not isinstance(output, dict):
        raise TypeError(
            f'{method}: unsupported predict output type {type(output).__name__}; '
            f'signature={summarize_value(output)}')

    for key in FINAL_OUTPUT_CANDIDATES:
        value = output.get(key)
        if torch.is_tensor(value):
            if key != 'x_final':
                warn(
                    f'{method}: output has no x_final; using tensor field '
                    f'"{key}" as the final prediction. '
                    f'output_signature={summarize_value(output)}')
            return value, key

    tensor_fields = [
        (key, value)
        for key, value in output.items()
        if torch.is_tensor(value) and value.dim() >= 2
    ]
    if len(tensor_fields) == 1:
        key, value = tensor_fields[0]
        warn(
            f'{method}: output has no known final-prediction field; using '
            f'the only tensor-like candidate "{key}". '
            f'output_signature={summarize_value(output)}')
        return value, key

    raise KeyError(
        f'{method}: cannot find final prediction tensor. Expected one of '
        f'{FINAL_OUTPUT_CANDIDATES}; output_signature={summarize_value(output)}')


def flatten_prediction_tensor(tensor, method):
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.dim() != 2:
        raise ValueError(
            f'{method}: final prediction must flatten to [B, N], got '
            f'{tuple(tensor.shape)}.')
    side = int(round(math.sqrt(tensor.shape[1])))
    if side * side != tensor.shape[1]:
        raise ValueError(
            f'{method}: final prediction length must be square, got '
            f'{tensor.shape[1]}.')
    return tensor, side


def skip_method(message):
    warn(message)
    raise MethodSkip(message)


def soft_thresholding(x, lambda_value):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_value, 0)


def ista(A, y, lambda_value, max_iter, tol, A_pinv, L):
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    x = np.dot(A_pinv, y)
    for _ in range(max_iter):
        x_new = soft_thresholding(
            x + A.T.dot(y - A.dot(x)) / L,
            lambda_value / L)
        if np.linalg.norm(x - x_new) < tol:
            return x_new.reshape(-1)
        x = x_new
    return x.reshape(-1)


def detections_from_map(matrix, metric_cfg, subpixel_offset=None):
    c = int(metric_cfg.get('c', 3))
    threshold = float(metric_cfg.get('brightness_threshold', 50))
    offset_limit = float(metric_cfg.get('subpixel_offset_limit', 0.5))
    center = (c - 1) // 2

    filtered = np.asarray(matrix, dtype=np.float32).copy()
    filtered[filtered < threshold] = 0
    rows, cols = np.nonzero(filtered)
    detections = []
    for row, col in zip(rows, cols):
        row_offset = 0.0
        col_offset = 0.0
        if subpixel_offset is not None:
            row_offset = float(np.clip(
                subpixel_offset[0, row, col], -offset_limit, offset_limit))
            col_offset = float(np.clip(
                subpixel_offset[1, row, col], -offset_limit, offset_limit))
        detections.append({
            'row': float((row + row_offset - center) / c),
            'col': float((col + col_offset - center) / c),
            'score': float(filtered[row, col]),
            'grid_row': int(row),
            'grid_col': int(col),
        })
    detections.sort(key=lambda item: item['score'], reverse=True)
    return detections, filtered


def detections_from_filtered_grid_without_offset(matrix, metric_cfg):
    c = int(metric_cfg.get('c', 3))
    threshold = float(metric_cfg.get('brightness_threshold', 50))
    center = (c - 1) // 2

    filtered = np.asarray(matrix, dtype=np.float32).copy()
    filtered[filtered < threshold] = 0
    rows, cols = np.nonzero(filtered)
    detections = []
    for row, col in zip(rows, cols):
        detections.append({
            'row': float((row - center) / c),
            'col': float((col - center) / c),
            'score': float(filtered[row, col]),
            'grid_row': int(row),
            'grid_col': int(col),
        })
    detections.sort(key=lambda item: item['score'], reverse=True)
    return detections


def reconstruction_grid_stats(gt_points, matrix, metric_cfg, distance_thr):
    detections = detections_from_filtered_grid_without_offset(
        matrix,
        metric_cfg)
    stats = localization_stats(gt_points, detections, distance_thr)
    return {
        'grid_recall': float(stats['recall']),
        'grid_precision': float(stats['precision']),
        'grid_mean_gt_distance': float(stats['mean_gt_distance']),
        'grid_det_count': int(stats['det_count']),
        'grid_matched': int(stats['matched']),
    }


def infer_neural_method(name, spec, samples, indices, args):
    config_path = as_path(spec['config'])
    checkpoint_path = as_path(spec['checkpoint'])
    if not config_path.exists():
        skip_method(f'{name}: config not found: {config_path}')
    if not checkpoint_path.exists():
        skip_method(f'{name}: checkpoint not found: {checkpoint_path}')
    if checkpoint_path.stat().st_size == 0 and not args.allow_empty_checkpoints:
        skip_method(f'{name}: checkpoint is empty: {checkpoint_path}')

    cfg = Config.fromfile(str(config_path))
    metric_cfg = resolve_metric_cfg(cfg, args)
    info(f'[{name}] config={spec["config"]}')
    info(f'[{name}] checkpoint={spec["checkpoint"]}')
    info(f'[{name}] model_type={cfg.model.get("type")}')

    if cfg.model.get('type') == 'ARLR':
        arlr_module = importlib.import_module('arlr.models.editor.arlr')
        arlr_module.device = torch.device(args.device)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(checkpoint_path), map_location=args.device)
    model.to(args.device)
    for attr_name in ('Phi', 'Qinit'):
        attr_value = getattr(model, attr_name, None)
        if torch.is_tensor(attr_value):
            setattr(model, attr_name, attr_value.to(args.device))
    model.eval()

    inference_batch_size = args.batch_size
    pad_partial_batch = False
    if cfg.model.get('dynamic_conv_module') == 'OriginalDynamicConv':
        # The original batch-coupled implementation creates one output
        # channel per sample, while the following layers expect 64 channels.
        # Preserve the batch size used by its training/evaluation config.
        inference_batch_size = int(
            cfg.get('test_dataloader', {}).get('batch_size', 64))
        pad_partial_batch = True
        if args.batch_size != inference_batch_size:
            warn(
                f'{name}: OriginalDynamicConv requires batch size '
                f'{inference_batch_size}; overriding requested batch size '
                f'{args.batch_size}.')

    results = {}
    runtime_info = {
        'config': spec.get('config'),
        'checkpoint': spec.get('checkpoint'),
        'resolved_config': str(config_path),
        'resolved_checkpoint': str(checkpoint_path),
        'model_type': cfg.model.get('type'),
        'metric_cfg': dict(metric_cfg),
        'subpixel_refine': bool(metric_cfg.get('subpixel_refine', False)),
        'subpixel_offset_in_output': False,
        'final_prediction_field': None,
        'output_signature': None,
        'requested_batch_size': int(args.batch_size),
        'inference_batch_size': int(inference_batch_size),
    }
    start = time.perf_counter()
    with torch.no_grad():
        for start_idx in range(0, len(samples), inference_batch_size):
            batch_samples = list(
                samples[start_idx:start_idx + inference_batch_size])
            batch_indices = indices[
                start_idx:start_idx + inference_batch_size]
            inference_samples = batch_samples
            if pad_partial_batch and len(batch_samples) < inference_batch_size:
                pad_count = inference_batch_size - len(batch_samples)
                inference_samples = (
                    batch_samples + [batch_samples[-1]] * pad_count)
                warn(
                    f'{name}: padded the final inference batch from '
                    f'{len(batch_samples)} to {inference_batch_size} samples '
                    'for OriginalDynamicConv compatibility.')
            raw_output = model(
                mode='predict', **make_batch(inference_samples))
            output = unwrap_predict_output(raw_output, name)
            output_signature = summarize_value(output)
            if runtime_info['output_signature'] is None:
                runtime_info['output_signature'] = output_signature
                info(f'[{name}] output_signature={output_signature}')

            final_tensor, final_field = extract_final_prediction_tensor(
                output, name)
            final_tensor, side = flatten_prediction_tensor(final_tensor, name)
            runtime_info['final_prediction_field'] = final_field
            if name == 'LISTA':
                message = (
                    f'[LISTA] final_prediction_field={final_field}; '
                    f'output_signature={output_signature}')
                if final_field != 'x_final':
                    warn(
                        'LISTA did not expose x_final; verify that the chosen '
                        f'field is the final reconstruction. {message}')
                else:
                    info(message)
            x_final = final_tensor.detach().float().cpu().numpy()
            maps = x_final.reshape(x_final.shape[0], side, side)

            offsets = None
            if metric_cfg.get('subpixel_refine', False):
                if not isinstance(output, dict) or 'subpixel_offset' not in output:
                    raise KeyError(
                        f'{name}: evaluator asks for subpixel_refine=True, '
                        'but model output has no subpixel_offset.')
                offsets = output['subpixel_offset'].detach().float().cpu().numpy()
                runtime_info['subpixel_offset_in_output'] = True

            for local_idx, sample_idx in enumerate(batch_indices):
                offset = offsets[local_idx] if offsets is not None else None
                dets, filtered_map = detections_from_map(
                    maps[local_idx], metric_cfg, offset)
                results[sample_idx] = {
                    'matrix': maps[local_idx].astype(np.float32),
                    'filtered_matrix': filtered_map.astype(np.float32),
                    'detections': dets,
                }
    elapsed = time.perf_counter() - start
    info(f'[{name}] inferred {len(results)} samples in {elapsed:.1f}s')
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results, metric_cfg, runtime_info


def infer_ista(samples, indices, cfg, args):
    metric_cfg = resolve_metric_cfg(cfg, args)
    metric_cfg['subpixel_refine'] = False
    phi_name = cfg.get('Phi_data_Name', 'data/sampling_matrix/a_phi_0_3.mat')
    phi_path = as_path(phi_name)
    if not phi_path.exists():
        skip_method(f'ISTA: phi matrix not found: {phi_path}')
    A = load_phi(phi_path)
    c = int(metric_cfg.get('c', 3))
    if A.shape != (121, (11 * c) ** 2):
        skip_method(f'ISTA: phi shape {A.shape} does not match c={c}.')
    A_pinv = np.linalg.pinv(A)
    L = np.linalg.norm(A, ord=2) ** 2
    results = {}
    start = time.perf_counter()
    for sample, sample_idx in zip(samples, indices):
        y = sample_to_numpy(sample, 'gt_img_11').reshape(-1, 1)
        x_output = ista(
            A, y, args.ista_lambda, args.ista_max_iter, args.ista_tol,
            A_pinv, L)
        matrix = x_output.reshape(11 * c, 11 * c)
        dets, filtered_map = detections_from_map(matrix, metric_cfg)
        results[sample_idx] = {
            'matrix': matrix.astype(np.float32),
            'filtered_matrix': filtered_map.astype(np.float32),
            'detections': dets,
        }
    elapsed = time.perf_counter() - start
    info(f'[ISTA] inferred {len(results)} samples in {elapsed:.1f}s')
    runtime_info = {
        'config': args.dataset_config,
        'checkpoint': None,
        'resolved_config': str(as_path(args.dataset_config)),
        'resolved_checkpoint': None,
        'model_type': 'traditional_ista',
        'metric_cfg': dict(metric_cfg),
        'subpixel_refine': False,
        'subpixel_offset_in_output': False,
        'final_prediction_field': 'ista_solution',
        'output_signature': {'type': 'ndarray', 'shape': [len(indices), (11 * c) ** 2]},
    }
    return results, metric_cfg, runtime_info


def read_gt(sample, is_mat=False):
    return read_targets_from_xml(sample['ann_path'], is_mat=is_mat)


def min_pair_distance(points):
    if len(points) < 2:
        return float('inf')
    coords = np.asarray([[p[0], p[1]] for p in points], dtype=np.float64)
    best = float('inf')
    for i in range(len(coords)):
        diff = coords[i + 1:] - coords[i]
        if diff.size:
            best = min(best, float(np.sqrt((diff * diff).sum(axis=1)).min()))
    return best


def greedy_matches(gt_points, detections, max_distance=float('inf')):
    pairs = []
    for gt_idx, gt in enumerate(gt_points):
        for det_idx, det in enumerate(detections):
            dist = math.hypot(float(gt[0]) - det['row'],
                              float(gt[1]) - det['col'])
            if dist <= max_distance:
                pairs.append((dist, gt_idx, det_idx))
    pairs.sort(key=lambda item: item[0])
    used_gt = set()
    used_det = set()
    matches = []
    for dist, gt_idx, det_idx in pairs:
        if gt_idx in used_gt or det_idx in used_det:
            continue
        used_gt.add(gt_idx)
        used_det.add(det_idx)
        matches.append((gt_idx, det_idx, dist))
    return matches


def localization_stats(gt_points, detections, distance_thr):
    gt_count = len(gt_points)
    det_count = len(detections)
    if gt_count == 0:
        return {
            'recall': 0.0,
            'precision': 0.0,
            'mean_gt_distance': float('inf'),
            'matched': 0,
            'det_count': det_count,
        }

    matches = greedy_matches(gt_points, detections, distance_thr)
    matched_gt = {item[0] for item in matches}
    nearest = []
    for gt_idx, gt in enumerate(gt_points):
        if det_count == 0:
            nearest.append(1.0)
            continue
        distances = [
            math.hypot(float(gt[0]) - det['row'], float(gt[1]) - det['col'])
            for det in detections
        ]
        best = min(distances)
        nearest.append(min(best, 1.0))
    return {
        'recall': len(matched_gt) / gt_count,
        'precision': len(matches) / det_count if det_count else 0.0,
        'mean_gt_distance': float(np.mean(nearest)),
        'matched': len(matches),
        'det_count': det_count,
    }


def normalized_map_difference(first_result, second_result,
                              keys=('matrix', 'filtered_matrix')):
    if first_result is None or second_result is None:
        return 0.0

    def clean_matrix(result, key):
        matrix = np.asarray(result.get(key, result.get('matrix')),
                            dtype=np.float32)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def normalize(matrix):
        matrix = matrix - float(matrix.min())
        peak = float(matrix.max())
        if peak > 0:
            matrix = matrix / peak
        return matrix

    scores = []
    for key in keys:
        first = normalize(clean_matrix(first_result, key))
        second = normalize(clean_matrix(second_result, key))
        if first.shape != second.shape:
            continue
        l1_diff = float(np.mean(np.abs(first - second)))
        first_flat = first.reshape(-1)
        second_flat = second.reshape(-1)
        if float(first_flat.std()) > 1e-8 and float(second_flat.std()) > 1e-8:
            corr = float(np.corrcoef(first_flat, second_flat)[0, 1])
            corr_diff = 0.5 * (1.0 - np.clip(corr, -1.0, 1.0))
        else:
            corr_diff = 0.0
        scores.append(0.65 * l1_diff + 0.35 * corr_diff)
    return float(np.mean(scores)) if scores else 0.0


def score_cases(samples_by_index, all_results, args):
    scored = []
    for sample_idx, sample in samples_by_index.items():
        gt_points = sample['gt_points']
        crowd_dist = min_pair_distance(gt_points)
        method_stats = {}
        for method, result_by_idx in all_results.items():
            if sample_idx not in result_by_idx:
                continue
            method_stats[method] = localization_stats(
                gt_points,
                result_by_idx[sample_idx]['detections'],
                args.distance_thr)
        if 'ARLR-Net' not in method_stats:
            continue

        arlr = method_stats['ARLR-Net']
        dista = method_stats.get('DISTA-Net')
        arlr_result = all_results.get('ARLR-Net', {}).get(sample_idx)
        dista_result = all_results.get('DISTA-Net', {}).get(sample_idx)
        arlr_dista_map_diff = normalized_map_difference(
            arlr_result,
            dista_result)
        arlr_dista_filtered_map_diff = normalized_map_difference(
            arlr_result,
            dista_result,
            keys=('filtered_matrix',))
        if dista is None:
            dista_recall = 0.0
            dista_err = 1.0
            arlr_recall = float(arlr['recall'])
            arlr_err = float(arlr['mean_gt_distance'])
            recall_gain = arlr_recall - dista_recall
            err_gain = dista_err - arlr_err
            advantage = -arlr['mean_gt_distance'] + arlr['recall']
        else:
            dista_recall = float(dista['recall'])
            dista_err = float(dista['mean_gt_distance'])
            arlr_recall = float(arlr['recall'])
            arlr_err = float(arlr['mean_gt_distance'])
            recall_gain = arlr_recall - dista_recall
            err_gain = dista_err - arlr_err
            advantage = (
                err_gain + 0.5 * recall_gain)
        crowd_bonus = 1.0 / (crowd_dist + 0.25) if np.isfinite(crowd_dist) else 0
        gt_count = len(gt_points)
        enough_crowding = gt_count >= 2 and crowd_dist <= args.max_crowd_distance
        count_bonus = 0.2 if gt_count >= 3 else 0.0
        improvement_bonus = (
            2.0 * max(recall_gain, 0.0) +
            1.4 * max(err_gain, 0.0))
        score = advantage + improvement_bonus + 0.15 * crowd_bonus + count_bonus
        if not enough_crowding:
            score -= 2.0
        scored.append({
            'sample_idx': sample_idx,
            'file_id': sample['file_id'],
            'gt_count': gt_count,
            'min_gt_pair_distance': crowd_dist,
            'score': score,
            'advantage': advantage,
            'dista_recall': dista_recall,
            'arlr_recall': arlr_recall,
            'dista_err': dista_err,
            'arlr_err': arlr_err,
            'arlr_precision': float(arlr['precision']),
            'arlr_det_count': int(arlr['det_count']),
            'arlr_filtered_det_count': int(arlr['det_count']),
            'arlr_filtered_det_count_gap': int(
                abs(int(arlr['det_count']) - gt_count)),
            'arlr_dista_map_diff': arlr_dista_map_diff,
            'arlr_dista_filtered_map_diff': arlr_dista_filtered_map_diff,
            'recall_gain': recall_gain,
            'err_gain': err_gain,
            'method_stats': method_stats,
        })
    scored.sort(key=lambda item: item['score'], reverse=True)
    return scored


def select_cases(scored_cases, args):
    rng = np.random.default_rng(args.random_seed)

    def crowded(case):
        return (
            case['gt_count'] >= 3
            and case['min_gt_pair_distance'] <= args.max_crowd_distance)

    def has_both_methods(case):
        return (
            'ARLR-Net' in case.get('method_stats', {})
            and 'DISTA-Net' in case.get('method_stats', {}))

    def soft_representative(case):
        return (
            crowded(case)
            and has_both_methods(case)
            and case['dista_recall'] >= args.min_dista_recall
            and case['dista_recall'] <= args.max_dista_recall
            and case['arlr_recall'] >= args.min_arlr_recall
            and (case['recall_gain'] > 0 or case['err_gain'] > 0))

    def shuffled(cases):
        cases = list(cases)
        if len(cases) > 1:
            order = rng.permutation(len(cases))
            cases = [cases[int(idx)] for idx in order]
        return cases

    def annotate(cases, source, from_balanced_pool=False):
        for case in cases:
            case['selection_source'] = source
            case['from_balanced_pool'] = bool(from_balanced_pool)
        return cases

    def balanced_selection():
        balanced_fallback = False
        if len(balanced_soft_pool) >= args.num_cases:
            selected_cases = annotate(
                shuffled(balanced_soft_pool)[:args.num_cases],
                'balanced_soft_random',
                from_balanced_pool=True)
            pool_name = 'balanced_soft_pool'
        elif len(balanced_pool) >= args.num_cases:
            selected_cases = annotate(
                shuffled(balanced_pool)[:args.num_cases],
                'balanced_crowded_random',
                from_balanced_pool=True)
            pool_name = 'balanced_crowded_pool'
            balanced_fallback = len(balanced_soft_pool) < args.num_cases
        else:
            selected_cases = annotate(
                advantage_ranked[:args.num_cases],
                'balanced_fallback_advantage',
                from_balanced_pool=False)
            pool_name = 'advantage_fallback'
            balanced_fallback = True
        return selected_cases, pool_name, balanced_fallback

    def arlr_success_score(case):
        det_over = max(0, case['arlr_det_count'] - case['gt_count'])
        det_gap = case.get(
            'arlr_filtered_det_count_gap',
            abs(case['arlr_det_count'] - case['gt_count']))
        det_close_bonus = 1.0 / (1.0 + det_gap)
        err = case['arlr_err']
        crowd_dist = case['min_gt_pair_distance']
        crowd_bonus = (
            1.0 / (crowd_dist + 0.25) if np.isfinite(crowd_dist) else 0.0)
        err_bonus = 0.35 if err <= 0.08 else 0.0
        return (
            6.0 * case['arlr_recall'] +
            2.0 * case['arlr_precision'] -
            4.0 * err -
            0.35 * det_over +
            0.12 * crowd_bonus +
            0.35 * det_close_bonus +
            err_bonus)

    def arlr_distinct_score(case):
        det_over = max(0, case['arlr_det_count'] - case['gt_count'])
        det_gap = case.get(
            'arlr_filtered_det_count_gap',
            abs(case['arlr_det_count'] - case['gt_count']))
        det_close_bonus = 1.0 / (1.0 + det_gap)
        crowd_dist = case['min_gt_pair_distance']
        crowd_bonus = (
            1.0 / (crowd_dist + 0.25) if np.isfinite(crowd_dist) else 0.0)
        score = (
            7.5 * case['arlr_recall'] +
            4.0 * case['arlr_precision'] -
            5.5 * case['arlr_err'] -
            0.75 * det_over +
            1.6 * det_close_bonus -
            0.35 * det_gap +
            0.35 * crowd_bonus +
            2.0 * case.get('arlr_dista_map_diff', 0.0) +
            4.0 * case.get('arlr_dista_filtered_map_diff', 0.0))
        method_stats = case.get('method_stats', {})
        dista = method_stats.get('DISTA-Net')
        if dista is not None:
            score += (
                9.0 * max(float(dista['mean_gt_distance']) - case['arlr_err'], 0.0) +
                8.0 * max(case['arlr_recall'] - float(dista['recall']), 0.0))
        other_bonuses = []
        for method, stats in method_stats.items():
            if method in ('ARLR-Net', 'DISTA-Net'):
                continue
            other_bonuses.append(
                1.2 * max(float(stats['mean_gt_distance']) - case['arlr_err'], 0.0) +
                1.0 * max(case['arlr_recall'] - float(stats['recall']), 0.0))
        if other_bonuses:
            score += float(np.mean(other_bonuses))
        return score

    def arlr_success_pool(min_recall, max_err, score_fn,
                           require_dista=False):
        pool = []
        for case in scored_cases:
            if int(case['gt_count']) not in RECON_TARGET_ROWS:
                continue
            if not crowded(case):
                continue
            if require_dista and not has_both_methods(case):
                continue
            if case['arlr_recall'] < min_recall:
                continue
            if case['arlr_err'] > max_err:
                continue
            pool.append(case)
        pool.sort(key=score_fn, reverse=True)
        return pool

    def diverse_arlr_selection(pool, source):
        selected_cases = []
        used = set()
        by_count = defaultdict(list)
        for case in pool:
            by_count[int(case['gt_count'])].append(case)
        for gt_count in RECON_TARGET_ROWS:
            for candidate in by_count.get(gt_count, []):
                if candidate['sample_idx'] not in used:
                    selected_cases.append(candidate)
                    used.add(candidate['sample_idx'])
                    break
        for candidate in pool:
            if len(selected_cases) >= args.num_cases:
                break
            if candidate['sample_idx'] in used:
                continue
            selected_cases.append(candidate)
            used.add(candidate['sample_idx'])
        return annotate(
            selected_cases[:args.num_cases],
            source,
            from_balanced_pool=False)

    def arlr_stage_specs(prefix):
        stages = []
        for source, min_recall, max_err in [
                (
                    f'{prefix}_preferred_err',
                    args.min_arlr_success_recall,
                    min(args.max_arlr_err, 0.08)),
                (
                    f'{prefix}_strict',
                    args.min_arlr_success_recall,
                    args.max_arlr_err),
                (
                    f'{prefix}_relaxed_err',
                    args.min_arlr_success_recall,
                    max(args.max_arlr_err, 0.15)),
                (
                    f'{prefix}_relaxed_recall',
                    min(args.min_arlr_success_recall, 0.75),
                    max(args.max_arlr_err, 0.15))]:
            if (min_recall, max_err) not in [
                    (item[1], item[2]) for item in stages]:
                stages.append((source, min_recall, max_err))
        return stages

    def run_arlr_staged_selection(prefix, score_fn, require_dista=False):
        for source, min_recall, max_err in arlr_stage_specs(prefix):
            pool = arlr_success_pool(
                min_recall,
                max_err,
                score_fn,
                require_dista=require_dista)
            arlr_success_pool_sizes[source] = len(pool)
            if len(pool) >= args.num_cases:
                return (
                    diverse_arlr_selection(pool, source),
                    source,
                    source != f'{prefix}_preferred_err',
                    {
                        'source': source,
                        'min_recall': min_recall,
                        'max_err': max_err,
                        'require_dista': bool(require_dista),
                    })
        return [], None, False, None

    advantage_preferred = [
        case for case in scored_cases
        if crowded(case)
        and case['dista_recall'] >= args.min_dista_recall
        and case['dista_recall'] <= args.max_dista_recall
        and case['arlr_recall'] >= args.min_arlr_recall
        and case['recall_gain'] >= args.min_recall_gain
        and case['err_gain'] >= args.min_error_gain
        and (case['recall_gain'] > 0 or case['err_gain'] > 0)
    ]
    advantage_positive = [
        case for case in scored_cases
        if case['gt_count'] >= 2
        and case['min_gt_pair_distance'] <= args.max_crowd_distance
        and case['advantage'] > 0
        and case not in advantage_preferred
    ]
    advantage_fallback = [
        case for case in scored_cases
        if case['gt_count'] >= 2
        and case['min_gt_pair_distance'] <= args.max_crowd_distance
        and case not in advantage_positive
        and case not in advantage_preferred
    ]
    advantage_remaining = [
        case for case in scored_cases
        if case not in advantage_preferred
        and case not in advantage_positive
        and case not in advantage_fallback
    ]
    advantage_ranked = (
        advantage_preferred + advantage_positive +
        advantage_fallback + advantage_remaining)

    balanced_pool = [
        case for case in scored_cases
        if crowded(case) and has_both_methods(case)
    ]
    balanced_soft_pool = [
        case for case in balanced_pool
        if soft_representative(case)
    ]

    all_both_methods = [
        case for case in scored_cases
        if has_both_methods(case)
    ]
    random_pool = (
        balanced_pool if args.random_crowded_only and balanced_pool
        else all_both_methods)

    fallback_used = False
    selection_pool_name = args.selection_mode
    arlr_success_pool_sizes = OrderedDict()
    arlr_success_stage = None
    if args.selection_mode == 'advantage':
        selected = annotate(
            advantage_ranked[:args.num_cases],
            'advantage',
            from_balanced_pool=False)
        fallback_used = len(advantage_preferred) < args.num_cases
        selection_pool_name = 'advantage_ranked'
    elif args.selection_mode == 'random':
        pool = random_pool if random_pool else all_both_methods
        fallback_used = len(pool) < args.num_cases
        selected = annotate(
            shuffled(pool)[:args.num_cases],
            'random_crowded' if args.random_crowded_only and balanced_pool
            else 'random_all',
            from_balanced_pool=False)
        selection_pool_name = (
            'random_crowded_pool' if args.random_crowded_only and balanced_pool
            else 'random_all_pool')
    elif args.selection_mode in (
            'arlr_success', 'arlr_distinct', 'recon_arlr_distinct'):
        if args.selection_mode in ('arlr_distinct', 'recon_arlr_distinct'):
            selected, selection_pool_name, fallback_used, arlr_success_stage = (
                run_arlr_staged_selection(
                    args.selection_mode,
                    arlr_distinct_score,
                    require_dista=True))
            if not selected:
                selected, selection_pool_name, fallback_used, arlr_success_stage = (
                    run_arlr_staged_selection(
                        args.selection_mode + '_no_dista_requirement',
                        arlr_distinct_score,
                        require_dista=False))
                if selected:
                    fallback_used = True
        else:
            selected, selection_pool_name, fallback_used, arlr_success_stage = (
                run_arlr_staged_selection(
                    'arlr_success',
                    arlr_success_score,
                    require_dista=False))
        if not selected:
            selected, selection_pool_name, fallback_used = balanced_selection()
            selected = annotate(
                selected,
                args.selection_mode + '_fallback_' + selection_pool_name,
                from_balanced_pool=selection_pool_name.startswith('balanced_'))
            fallback_used = True
            arlr_success_stage = {
                'source': args.selection_mode + '_fallback_balanced',
                'min_recall': None,
                'max_err': None,
                'require_dista': None,
            }
    else:
        selected, selection_pool_name, fallback_used = balanced_selection()

    for case in selected:
        case['selection_mode'] = args.selection_mode
        case['fallback_used'] = bool(fallback_used)

    selection_info = {
        'selection_mode': args.selection_mode,
        'random_seed': args.random_seed,
        'random_crowded_only': bool(args.random_crowded_only),
        'fallback_used': bool(fallback_used),
        'selection_pool_name': selection_pool_name,
        'max_arlr_err': args.max_arlr_err,
        'min_arlr_success_recall': args.min_arlr_success_recall,
        'arlr_success_pool_sizes': arlr_success_pool_sizes,
        'arlr_success_stage': arlr_success_stage,
        'num_advantage_preferred_cases': len(advantage_preferred),
        'num_balanced_soft_cases': len(balanced_soft_pool),
        'num_balanced_crowded_cases': len(balanced_pool),
        'num_random_pool_cases': len(random_pool),
        'num_selected_from_balanced_pool': sum(
            bool(case.get('from_balanced_pool', False)) for case in selected),
        'min_recall_gain': args.min_recall_gain,
        'min_error_gain': args.min_error_gain,
        'min_dista_recall': args.min_dista_recall,
        'max_dista_recall': args.max_dista_recall,
        'min_arlr_recall': args.min_arlr_recall,
        'preferred_rule': (
            'gt_count >= 3, crowded, min_dista_recall <= dista_recall '
            '<= max_dista_recall, arlr_recall >= min_arlr_recall, '
            'recall_gain >= min_recall_gain, err_gain >= min_error_gain, '
            'and recall_gain > 0 or err_gain > 0'),
        'mode_descriptions': {
            'balanced': (
                'Representative crowded cases selected with fixed random seed.'),
            'random': (
                'Randomly selected test cases with fixed random seed.'),
            'advantage': (
                'Cases selected by ARLR-Net localization advantage.'),
            'arlr_success': (
                'Successful ARLR-Net crowded c=3/4/5 cases selected for '
                'paper reconstruction and localization figures.'),
            'arlr_distinct': (
                'Successful ARLR-Net crowded c=3/4/5 cases ranked to prefer '
                'larger reconstruction/localization separation from DISTA-Net '
                'and other available comparison methods.'),
            'recon_arlr_distinct': (
                'Uses ARLR-vs-DISTA distinct localization selection for '
                'localization figures, while reconstruction_comparison_grid '
                'uses stricter reconstruction-specific ARLR-correct and '
                'DISTA-distinct row selection.'),
        },
    }
    if not selected:
        warn('No suitable cases were selected; figures will not be generated.')
    elif selection_info['fallback_used']:
        warn(
            'Case selection used a fallback pool because the preferred pool '
            'was too small for the requested number of cases.')
    return selected, selection_info


def collect_matched_coords(gt_points, result, distance_thr):
    coords = [[float(point[0]), float(point[1])] for point in gt_points]
    if result is None:
        return coords
    detections = result.get('detections', [])
    for _, det_idx, _ in greedy_matches(gt_points, detections, distance_thr):
        if det_idx < len(detections):
            det = detections[det_idx]
            coords.append([det['row'], det['col']])
    return coords


def zoom_from_coords(coords, pad=0.9, min_size=2.2):
    if not coords:
        return (PLOT_DOMAIN[0], PLOT_DOMAIN[1], PLOT_DOMAIN[0], PLOT_DOMAIN[1])

    coords = np.asarray(coords, dtype=np.float64)
    row_min, col_min = coords.min(axis=0)
    row_max, col_max = coords.max(axis=0)
    row_center = (row_min + row_max) / 2.0
    col_center = (col_min + col_max) / 2.0
    height = max(row_max - row_min + 2 * pad, min_size)
    width = max(col_max - col_min + 2 * pad, min_size)
    size = min(max(height, width), PLOT_DOMAIN[1] - PLOT_DOMAIN[0])

    def bounds(center):
        lower = center - size / 2.0
        upper = center + size / 2.0
        if lower < PLOT_DOMAIN[0]:
            upper += PLOT_DOMAIN[0] - lower
            lower = PLOT_DOMAIN[0]
        if upper > PLOT_DOMAIN[1]:
            lower -= upper - PLOT_DOMAIN[1]
            upper = PLOT_DOMAIN[1]
        lower = max(PLOT_DOMAIN[0], lower)
        upper = min(PLOT_DOMAIN[1], upper)
        return lower, upper

    xmin, xmax = bounds(col_center)
    ymin, ymax = bounds(row_center)
    return xmin, xmax, ymin, ymax


def zoom_for_case(gt_points, result_by_method, methods, distance_thr, pad=0.9):
    coords = []
    for point in gt_points:
        coords.append([float(point[0]), float(point[1])])
    for method in methods:
        result = result_by_method.get(method)
        coords.extend(collect_matched_coords(gt_points, result, distance_thr))
    return zoom_from_coords(coords, pad=pad)


def aligned_localization_zoom(gt_points, result_by_method, distance_thr):
    return zoom_for_case(
        gt_points,
        result_by_method,
        ('DISTA-Net', 'ARLR-Net'),
        distance_thr,
        pad=0.72)


def matrix_crop_from_zoom(matrix, zoom):
    if zoom is None:
        return np.asarray(matrix, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    height, width = matrix.shape
    c = max(1, int(round(height / 11)))
    center = (c - 1) // 2
    x_min, x_max, y_min, y_max = zoom
    col_min = max(0, int(math.floor(x_min * c + center - 1)))
    col_max = min(width, int(math.ceil(x_max * c + center + 2)))
    row_min = max(0, int(math.floor(y_min * c + center - 1)))
    row_max = min(height, int(math.ceil(y_max * c + center + 2)))
    if row_min >= row_max or col_min >= col_max:
        return matrix
    return matrix[row_min:row_max, col_min:col_max]


def intensity_limits_for_case(results, gt_matrix=None):
    values = []
    if gt_matrix is not None:
        gt_values = np.asarray(gt_matrix, dtype=np.float32)
        values.append(gt_values[gt_values > 0])
    for result in results:
        if result is None:
            continue
        matrix = np.asarray(result.get('filtered_matrix'), dtype=np.float32)
        positives = matrix[matrix > 0]
        if positives.size:
            values.append(positives)
    values = [value for value in values if value.size]
    if not values:
        return None
    merged = np.concatenate(values)
    vmax = float(np.percentile(merged, 99.2))
    if vmax <= 0:
        vmax = float(merged.max())
    return 0.0, max(vmax, 1e-6)


def black_floor_cmap(name):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad('black')
    cmap.set_under('black')
    return cmap


def black_floor_matrix(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    return np.ma.masked_where(matrix <= 0, matrix)


def plot_points(ax, gt_points, detections=None, matches=None, title=None,
                matrix=None, plot_topk=80, zoom=None, hide_ticks=False,
                intensity_limits=None, gt_size=58, pred_size_range=(18, 64),
                line_width=0.9, matched_only=False, background_cmap='gray',
                background_alpha=0.62, show_unmatched_faint=False,
                show_error_lines=False):
    ax.set_facecolor('black')
    if matrix is not None:
        c = max(1, int(round(matrix.shape[0] / 11)))
        center = (c - 1) // 2
        min_coord = (-center - 0.5) / c
        max_coord = (matrix.shape[0] - 1 - center + 0.5) / c
        norm = None
        if intensity_limits is not None:
            norm = mcolors.PowerNorm(
                gamma=0.62,
                vmin=float(intensity_limits[0]),
                vmax=float(intensity_limits[1]))
        ax.imshow(
            black_floor_matrix(matrix),
            cmap=black_floor_cmap(background_cmap),
            origin='lower',
            extent=(min_coord, max_coord, min_coord, max_coord),
            interpolation='nearest',
            alpha=background_alpha,
            norm=norm)
    else:
        ax.set_facecolor(PANEL_FACE_COLOR)

    gt_arr = np.asarray([[p[0], p[1]] for p in gt_points], dtype=np.float64)
    if gt_arr.size:
        ax.scatter(
            gt_arr[:, 1], gt_arr[:, 0],
            s=gt_size,
            marker='x',
            linewidths=1.55,
            c=GT_COLOR,
            label='GT',
            zorder=5)

    detections = detections or []
    matches = matches or []
    unmatched_detections = []
    if matched_only:
        matched_indices = []
        for _, det_idx, _ in matches:
            if det_idx < len(detections) and det_idx not in matched_indices:
                matched_indices.append(det_idx)
        matched_index_set = set(matched_indices)
        unmatched_detections = [
            det for idx, det in enumerate(detections)
            if idx not in matched_index_set
        ]
        index_map = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(matched_indices)
        }
        detections = [detections[idx] for idx in matched_indices]
        matches = [
            (gt_idx, index_map[det_idx], dist)
            for gt_idx, det_idx, dist in matches
            if det_idx in index_map
        ]
    if matched_only and show_unmatched_faint and unmatched_detections:
        faint_dets = (
            unmatched_detections[:plot_topk]
            if plot_topk > 0 else unmatched_detections)
        faint_arr = np.asarray(
            [[d['row'], d['col']] for d in faint_dets],
            dtype=np.float64)
        ax.scatter(
            faint_arr[:, 1], faint_arr[:, 0],
            s=12,
            marker='o',
            c='#8FB9C8',
            alpha=0.15,
            linewidths=0,
            label='Unmatched prediction',
            zorder=3)
    plot_dets = detections[:plot_topk] if plot_topk > 0 else detections
    if plot_dets:
        det_arr = np.asarray([[d['row'], d['col'], d['score']] for d in plot_dets])
        min_size, max_size = pred_size_range
        sizes = min_size + (max_size - min_size) * (
            (det_arr[:, 2] - det_arr[:, 2].min()) /
            max(float(np.ptp(det_arr[:, 2])), 1e-6))
        ax.scatter(
            det_arr[:, 1], det_arr[:, 0],
            s=sizes,
            facecolors='none',
            edgecolors=PRED_COLOR,
            linewidths=1.8,
            label='Prediction',
            zorder=6)

    if show_error_lines and matches:
        for gt_idx, det_idx, _ in matches:
            if gt_idx >= len(gt_points) or det_idx >= len(detections):
                continue
            gt = gt_points[gt_idx]
            det = detections[det_idx]
            con = ConnectionPatch(
                xyA=(gt[1], gt[0]),
                xyB=(det['col'], det['row']),
                coordsA='data',
                coordsB='data',
                axesA=ax,
                axesB=ax,
                color=ERROR_LINE_COLOR,
                linewidth=line_width,
                alpha=0.82,
                zorder=4)
            ax.add_artist(con)

    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal', adjustable='box')
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(labelsize=8, colors='#374151')
    for spine in ax.spines.values():
        spine.set_color('#CBD5E1')
        spine.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=10, pad=5)


def crowded_zoom(gt_points):
    if len(gt_points) < 2:
        return None
    coords = np.asarray([[p[0], p[1]] for p in gt_points], dtype=np.float64)
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if best is None or dist < best[0]:
                best = (dist, i, j)
    pair = coords[[best[1], best[2]]]
    row_center, col_center = pair.mean(axis=0)
    radius = max(0.9, best[0] * 0.85)
    return (
        max(-0.5, col_center - radius),
        min(10.5, col_center + radius),
        max(-0.5, row_center - radius),
        min(10.5, row_center + radius),
    )


def save_figure(fig, out_dir, stem):
    png_path = out_dir / f'{stem}.png'
    pdf_path = out_dir / f'{stem}.pdf'
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches='tight',
        facecolor=fig.get_facecolor())
    fig.savefig(
        pdf_path,
        bbox_inches='tight',
        facecolor=fig.get_facecolor())
    plt.close(fig)
    info(f'Saved {png_path}')
    info(f'Saved {pdf_path}')


def add_localization_legend(fig, y=0.02, show_error_lines=False):
    handles = [
        Line2D(
            [0], [0],
            marker='x',
            color='none',
            markeredgecolor=GT_COLOR,
            markeredgewidth=2.2,
            markersize=9,
            label='Yellow ×: Ground truth'),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor='none',
            markeredgecolor=PRED_COLOR,
            markeredgewidth=1.8,
            markersize=8,
            label='Cyan circle: Matched prediction'),
        Line2D(
            [0], [0],
            color='#7C3AED',
            linewidth=4,
            alpha=0.75,
            label='Colored background: Final intensity map'),
    ]
    handles[0].set_label('Yellow x: Ground truth')
    handles[0].set_markeredgewidth(1.6)
    if show_error_lines:
        handles.insert(
            2,
            Line2D(
                [0], [0],
                color=ERROR_LINE_COLOR,
                linewidth=1.3,
                label='White line: Localization error'))
    legend = fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=True,
        fontsize=7.8,
        handlelength=1.8,
        columnspacing=1.0,
        borderpad=0.35,
        labelspacing=0.3)
    legend.get_frame().set_facecolor('#111827')
    legend.get_frame().set_edgecolor('#334155')
    legend.get_frame().set_alpha(0.92)
    for text in legend.get_texts():
        text.set_color('white')


def draw_arlr_vs_dista(selected, samples_by_index, all_results, out_dir, args):
    if 'ARLR-Net' not in all_results or 'DISTA-Net' not in all_results:
        warn('ARLR-Net or DISTA-Net predictions are unavailable; skipping '
             'ARLR-vs-DISTA case figure.')
        return

    rows = len(selected)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(12.8, max(3.0, rows * 2.8)),
        squeeze=False)
    fig.suptitle(
        'Localization comparison on automatically selected crowded CSO test cases',
        fontsize=14,
        y=1.01)
    col_titles = ['LR input', 'GT points', 'DISTA-Net final output',
                  'ARLR-Net final output']

    for row_idx, case in enumerate(selected):
        sample_idx = case['sample_idx']
        sample = samples_by_index[sample_idx]
        gt_points = sample['gt_points']
        lr = sample['lr_image']
        gt_matrix = sample['gt_matrix']
        zoom = crowded_zoom(gt_points)

        ax = axes[row_idx, 0]
        ax.imshow(lr, cmap='gray', origin='lower', interpolation='nearest')
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
        ax.tick_params(labelsize=8)
        ax.set_title(col_titles[0] if row_idx == 0 else '', fontsize=10)
        ax.set_ylabel(f'#{sample_idx}\nfile {case["file_id"]}', fontsize=9)

        plot_points(
            axes[row_idx, 1],
            gt_points,
            title=col_titles[1] if row_idx == 0 else '',
            matrix=gt_matrix,
            plot_topk=args.plot_topk_dets,
            zoom=zoom)

        for col_idx, method in enumerate(('DISTA-Net', 'ARLR-Net'), start=2):
            result = all_results[method][sample_idx]
            matches = greedy_matches(
                gt_points, result['detections'], args.distance_thr)
            stats = localization_stats(
                gt_points, result['detections'], args.distance_thr)
            title = col_titles[col_idx] if row_idx == 0 else ''
            subtitle = (
                f'R@{args.distance_thr:g}={stats["recall"]:.2f}, '
                f'err={stats["mean_gt_distance"]:.3f}')
            if title:
                title = f'{title}\n{subtitle}'
            else:
                title = subtitle
            plot_points(
                axes[row_idx, col_idx],
                gt_points,
                result['detections'],
                matches,
                title=title,
                matrix=result['filtered_matrix'],
                plot_topk=args.plot_topk_dets,
                zoom=zoom)

    handles, labels = axes[0, 2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(pad=1.2)
    save_figure(fig, out_dir, 'arlr_vs_dista_localization_cases')


def draw_arlr_dista_zoom_cases(selected, samples_by_index, all_results, out_dir,
                                args):
    if 'ARLR-Net' not in all_results or 'DISTA-Net' not in all_results:
        warn('ARLR-Net or DISTA-Net predictions are unavailable; skipping '
             'arlr_dista_zoom_cases.')
        return

    rows = len(selected)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(9.2, max(2.75, rows * 2.55)),
        squeeze=False)
    fig.suptitle(
        'Zoomed Localization Comparison',
        fontsize=14,
        y=1.01)
    col_titles = ['GT', 'DISTA-Net', 'ARLR-Net']

    for row_idx, case in enumerate(selected):
        sample_idx = case['sample_idx']
        sample = samples_by_index[sample_idx]
        gt_points = sample['gt_points']
        result_by_method = {
            method: all_results[method].get(sample_idx)
            for method in ('DISTA-Net', 'ARLR-Net')
        }
        zoom = zoom_for_case(
            gt_points,
            result_by_method,
            ('DISTA-Net', 'ARLR-Net'),
            args.distance_thr,
            pad=0.95)
        limits = intensity_limits_for_case(
            [result_by_method['DISTA-Net'], result_by_method['ARLR-Net']],
            sample['gt_matrix'])

        for col_idx, method in enumerate(col_titles):
            ax = axes[row_idx, col_idx]
            title = method if row_idx == 0 else ''
            if method == 'GT':
                matrix = None if args.hide_intensity_background else sample['gt_matrix']
                plot_points(
                    ax,
                    gt_points,
                    title=title,
                    matrix=matrix,
                    plot_topk=args.plot_topk_dets,
                    zoom=zoom,
                    hide_ticks=True,
                    intensity_limits=limits,
                    gt_size=76,
                    pred_size_range=(56, 96),
                    line_width=1.05,
                    matched_only=args.plot_matched_only,
                    background_cmap=args.background_cmap,
                    background_alpha=args.background_alpha,
                    show_unmatched_faint=args.show_unmatched_faint,
                    show_error_lines=(
                        args.show_error_lines
                        and not args.clean_localization_style))
                continue

            result = result_by_method[method]
            matches = greedy_matches(
                gt_points, result['detections'], args.distance_thr)
            stats = localization_stats(
                gt_points, result['detections'], args.distance_thr)
            subtitle = f'R={stats["recall"]:.2f}, err={stats["mean_gt_distance"]:.3f}'
            if title:
                title = f'{title}\n{subtitle}'
            else:
                title = subtitle
            plot_points(
                ax,
                gt_points,
                result['detections'],
                matches,
                title=title,
                matrix=(None if args.hide_intensity_background
                        else result['filtered_matrix']),
                plot_topk=args.plot_topk_dets,
                zoom=zoom,
                hide_ticks=True,
                intensity_limits=limits,
                gt_size=76,
                pred_size_range=(56, 96),
                line_width=1.05,
                matched_only=args.plot_matched_only,
                background_cmap=args.background_cmap,
                background_alpha=args.background_alpha,
                show_unmatched_faint=args.show_unmatched_faint,
                show_error_lines=(
                    args.show_error_lines
                    and not args.clean_localization_style))
            if method == 'ARLR-Net':
                ax.title.set_fontweight('bold')
                ax.title.set_color('#1F2937')

    fig.tight_layout(pad=0.68, rect=(0.0, 0.08, 1.0, 0.96))
    add_localization_legend(
        fig,
        y=0.012,
        show_error_lines=(
            args.show_error_lines and not args.clean_localization_style))
    save_figure(fig, out_dir, 'arlr_dista_zoom_cases')


def draw_all_methods_summary(selected, all_results, out_dir, args):
    summary = []
    for method, result_by_idx in all_results.items():
        recalls = []
        distances = []
        precisions = []
        det_counts = []
        for case in selected:
            sample_idx = case['sample_idx']
            if sample_idx not in result_by_idx:
                continue
            gt_points = case['gt_points']
            stats = localization_stats(
                gt_points, result_by_idx[sample_idx]['detections'],
                args.distance_thr)
            recalls.append(stats['recall'])
            precisions.append(stats['precision'])
            distances.append(stats['mean_gt_distance'])
            det_counts.append(stats['det_count'])
        if recalls:
            summary.append({
                'method': method,
                'mean_recall': float(np.mean(recalls)),
                'mean_precision': float(np.mean(precisions)),
                'mean_gt_distance': float(np.mean(distances)),
                'mean_det_count': float(np.mean(det_counts)),
            })

    summary.sort(key=lambda item: item['mean_gt_distance'])
    methods = [item['method'] for item in summary]
    y = np.arange(len(methods))
    colors = [
        HIGHLIGHT_COLORS.get(method, DEFAULT_COLOR)
        for method in methods
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.8, max(4.0, len(methods) * 0.38)),
        gridspec_kw={'width_ratios': [1.2, 1.0]})
    distances = [item['mean_gt_distance'] for item in summary]
    recalls = [item['mean_recall'] for item in summary]
    precisions = [item['mean_precision'] for item in summary]

    ax1.barh(y, distances, color=colors, alpha=0.88, height=0.68)
    ax1.set_yticks(y, labels=methods)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean nearest-GT localization error on selected cases')
    ax1.grid(axis='x', linestyle='--', alpha=0.25)
    ax1.set_title('Lower is better', fontsize=11)

    ax2.scatter(recalls, y, s=70, c=colors, label='Recall', zorder=3)
    ax2.scatter(precisions, y, s=42, c='white', edgecolors=colors,
                linewidths=1.4, label='Precision', zorder=3)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_yticks(y, labels=[])
    ax2.invert_yaxis()
    ax2.set_xlabel(f'Point match at distance <= {args.distance_thr:g}')
    ax2.grid(axis='x', linestyle='--', alpha=0.25)
    ax2.legend(frameon=False, loc='lower right')
    ax2.set_title('Detection quality on the same cases', fontsize=11)

    fig.suptitle(
        'All-method localization summary for the selected crowded CSO test cases',
        fontsize=14)
    fig.tight_layout()
    save_figure(fig, out_dir, 'all_methods_localization_summary')
    return summary


def draw_all_methods_case_grid(selected, samples_by_index, all_results, out_dir,
                               args):
    if not selected:
        return
    case = selected[0]
    sample_idx = case['sample_idx']
    sample = samples_by_index[sample_idx]
    gt_points = sample['gt_points']
    methods = [method for method in METHOD_SPECS if method in all_results]
    panels = ['GT'] + methods
    cols = 4
    rows = int(math.ceil(len(panels) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(13.0, max(3.2, rows * 3.0)), squeeze=False)
    fig.suptitle(
        f'Final localization outputs on selected case '
        f'#{sample_idx} / file {case["file_id"]}',
        fontsize=14)
    result_by_method = {
        method: all_results[method].get(sample_idx)
        for method in methods
    }
    zoom = zoom_for_case(
        gt_points,
        result_by_method,
        methods,
        args.distance_thr,
        pad=0.95)
    limits = intensity_limits_for_case(
        [result_by_method.get(method) for method in methods],
        sample['gt_matrix'])

    flat_axes = axes.ravel()
    plot_points(
        flat_axes[0],
        gt_points,
        title='GT',
        matrix=(None if args.hide_intensity_background else sample['gt_matrix']),
        plot_topk=args.plot_topk_dets,
        zoom=zoom,
        hide_ticks=True,
        intensity_limits=limits,
        gt_size=82,
        pred_size_range=(38, 78),
        line_width=1.05,
        matched_only=True,
        background_cmap=args.background_cmap,
        background_alpha=args.background_alpha,
        show_unmatched_faint=args.show_unmatched_faint,
        show_error_lines=args.show_error_lines)
    for ax, method in zip(flat_axes[1:], methods):
        result = result_by_method.get(method)
        if result is None:
            ax.axis('off')
            continue
        matches = greedy_matches(gt_points, result['detections'],
                                 args.distance_thr)
        stats = localization_stats(gt_points, result['detections'],
                                   args.distance_thr)
        title = (
            f'{method}\nR={stats["recall"]:.2f}, '
            f'err={stats["mean_gt_distance"]:.3f}')
        plot_points(
            ax,
            gt_points,
            result['detections'],
            matches,
            title=title,
            matrix=(None if args.hide_intensity_background
                    else result['filtered_matrix']),
            plot_topk=args.plot_topk_dets,
            zoom=zoom,
            hide_ticks=True,
            intensity_limits=limits,
            gt_size=82,
            pred_size_range=(38, 78),
            line_width=1.05,
            matched_only=True,
            background_cmap=args.background_cmap,
            background_alpha=args.background_alpha,
            show_unmatched_faint=args.show_unmatched_faint,
            show_error_lines=args.show_error_lines)
    for ax in flat_axes[len(panels):]:
        ax.axis('off')
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.94))
    add_localization_legend(
        fig, y=0.012, show_error_lines=args.show_error_lines)
    save_figure(fig, out_dir, 'all_methods_best_case_grid')


def matrix_for_reconstruction_panel(method, sample_idx, samples_by_index,
                                    all_results, recon_map_type='filtered',
                                    warnings_out=None,
                                    fallback_warning_keys=None):
    if method == 'GT':
        return np.asarray(samples_by_index[sample_idx]['gt_matrix'],
                          dtype=np.float32)
    if method not in all_results or sample_idx not in all_results[method]:
        return None
    result = all_results[method][sample_idx]
    key = 'filtered_matrix' if recon_map_type == 'filtered' else 'matrix'
    if key not in result:
        if recon_map_type == 'filtered' and 'matrix' in result:
            warning_key = (method, sample_idx, key)
            if (fallback_warning_keys is None
                    or warning_key not in fallback_warning_keys):
                message = (
                    f'Reconstruction grid: {method} sample {sample_idx} has '
                    'no filtered_matrix; falling back to raw matrix.')
                warn(message)
                if warnings_out is not None:
                    warnings_out.append(message)
                if fallback_warning_keys is not None:
                    fallback_warning_keys.add(warning_key)
            key = 'matrix'
        else:
            return None
    return np.asarray(
        result[key],
        dtype=np.float32)


def reconstruction_crop_bounds(sample, matrix_shape, pad=5, min_size=23):
    gt_points = sample.get('gt_points', [])
    if not gt_points:
        return None
    height, width = matrix_shape
    c = max(1, int(round(height / 11)))
    center = (c - 1) // 2
    rows = [float(point[0]) * c + center for point in gt_points]
    cols = [float(point[1]) * c + center for point in gt_points]
    row_min = math.floor(min(rows)) - pad
    row_max = math.ceil(max(rows)) + pad + 1
    col_min = math.floor(min(cols)) - pad
    col_max = math.ceil(max(cols)) + pad + 1

    def square_bounds(low, high, size, target_span):
        center_pos = 0.5 * (low + high)
        low = int(math.floor(center_pos - target_span / 2))
        high = low + target_span
        if low < 0:
            high -= low
            low = 0
        if high > size:
            low -= high - size
            high = size
        return max(0, low), min(size, high)

    target_span = int(max(min_size, row_max - row_min, col_max - col_min))
    target_span = min(target_span, height, width)
    row_min, row_max = square_bounds(row_min, row_max, height, target_span)
    col_min, col_max = square_bounds(col_min, col_max, width, target_span)
    if row_min == 0 and row_max == height and col_min == 0 and col_max == width:
        return None
    return int(row_min), int(row_max), int(col_min), int(col_max)


def crop_matrix_for_display(matrix, crop_bounds):
    if crop_bounds is None:
        return matrix
    row_min, row_max, col_min, col_max = crop_bounds
    return matrix[row_min:row_max, col_min:col_max]


def matrix_extent(matrix_shape, crop_bounds=None):
    height, _ = matrix_shape
    c = max(1, int(round(height / 11)))
    center = (c - 1) // 2
    if crop_bounds is None:
        row_min = 0
        row_max = matrix_shape[0]
        col_min = 0
        col_max = matrix_shape[1]
    else:
        row_min, row_max, col_min, col_max = crop_bounds
    x_min = (col_min - center - 0.5) / c
    x_max = (col_max - center - 0.5) / c
    y_min = (row_min - center - 0.5) / c
    y_max = (row_max - center - 0.5) / c
    return x_min, x_max, y_min, y_max


def row_vmax_for_reconstruction(sample_idx, samples_by_index, all_results,
                                methods, crop_bounds=None,
                                recon_map_type='filtered',
                                warnings_out=None,
                                fallback_warning_keys=None):
    values = []
    for method in methods:
        matrix = matrix_for_reconstruction_panel(
            method,
            sample_idx,
            samples_by_index,
            all_results,
            recon_map_type,
            warnings_out,
            fallback_warning_keys)
        if matrix is None:
            continue
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = crop_matrix_for_display(matrix, crop_bounds)
        positive = matrix[matrix > 0]
        if positive.size:
            values.append(positive)
    if not values:
        return 1.0
    merged = np.concatenate(values)
    vmax = float(np.percentile(merged, 99.5))
    if vmax <= 0:
        vmax = float(merged.max())
    return max(vmax, 1e-6)


def reconstruction_label_position(matrix):
    matrix = np.nan_to_num(np.asarray(matrix, dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.size == 0 or float(matrix.max()) <= 0:
        return dict(x=0.04, y=0.94, ha='left', va='top')

    height, width = matrix.shape
    threshold = max(
        float(np.percentile(matrix, 92.0)),
        float(matrix.max()) * 0.18)
    focus = np.where(matrix >= threshold, matrix, 0.0)
    if float(focus.sum()) <= 0:
        focus = matrix

    candidates = [
        ('left_bottom', 0.04, 0.06, 'left', 'bottom', 0.00, 0.44, 0.00, 0.22),
        ('right_bottom', 0.96, 0.06, 'right', 'bottom', 0.56, 1.00, 0.00, 0.22),
        ('left_top', 0.04, 0.94, 'left', 'top', 0.00, 0.44, 0.78, 1.00),
        ('right_top', 0.96, 0.94, 'right', 'top', 0.56, 1.00, 0.78, 1.00),
    ]
    best = None
    for name, x, y, ha, va, col0, col1, row0, row1 in candidates:
        c0 = int(math.floor(col0 * width))
        c1 = max(c0 + 1, int(math.ceil(col1 * width)))
        r0 = int(math.floor(row0 * height))
        r1 = max(r0 + 1, int(math.ceil(row1 * height)))
        patch = focus[r0:r1, c0:c1]
        patch_energy = float(patch.sum())
        if 'bottom' in name:
            patch_energy *= 1.08
        if best is None or patch_energy < best[0]:
            best = (patch_energy, x, y, ha, va)
    _, x, y, ha, va = best
    return dict(x=x, y=y, ha=ha, va=va)


def reconstruction_methods_and_warnings(all_results, recon_methods):
    methods = []
    warnings_out = []
    for method in recon_methods:
        if method == 'GT':
            methods.append(method)
        elif method in all_results:
            methods.append(method)
        else:
            message = (
                f'Reconstruction grid: {method} has no successful inference '
                'result and is skipped.')
            warn(message)
            warnings_out.append(message)
    if 'GT' not in methods:
        methods.insert(0, 'GT')
    return methods, warnings_out


def reconstruction_dista_distinct_metrics(candidate):
    gt_count = int(candidate['gt_count'])
    err_thr = RECON_DISTINCT_ERR_GAIN_THRESHOLDS.get(gt_count, 0.020)
    diff_thr = RECON_DISTINCT_FILTERED_DIFF_THRESHOLDS.get(gt_count, 0.035)
    grid_err_gain = (
        candidate['dista_grid_err'] - candidate['arlr_grid_err'])
    grid_recall_gain = (
        candidate['arlr_grid_recall'] - candidate['dista_grid_recall'])
    det_gap_advantage = max(
        candidate['dista_grid_det_count_gap']
        - candidate['arlr_grid_det_count_gap'],
        0)
    filtered_diff = candidate['arlr_dista_filtered_map_diff']
    distinct_conditions = [
        grid_err_gain >= err_thr,
        filtered_diff >= diff_thr,
        grid_recall_gain > 0.0,
        det_gap_advantage > 0,
    ]
    condition_count = sum(1 for item in distinct_conditions if item)
    distinct_score = (
        7.0 * max(grid_err_gain, 0.0) / max(err_thr, 1e-6) +
        6.0 * max(filtered_diff, 0.0) / max(diff_thr, 1e-6) +
        8.0 * max(grid_recall_gain, 0.0) +
        5.0 * det_gap_advantage)
    return {
        'grid_err_gain': float(grid_err_gain),
        'grid_recall_gain': float(grid_recall_gain),
        'grid_det_count_gap_advantage': int(det_gap_advantage),
        'dista_distinct_score': float(distinct_score),
        'dista_distinct_condition_count': int(condition_count),
        'dista_distinct_strong': bool(condition_count > 0),
        'dista_distinct_err_gain_threshold': err_thr,
        'dista_distinct_filtered_diff_threshold': diff_thr,
    }


def reconstruction_candidate_record(sample_idx, sample, all_results, args,
                                    metric_cfg):
    gt_points = sample['gt_points']
    gt_count = len(gt_points)
    arlr_result = all_results.get('ARLR-Net', {}).get(sample_idx)
    if arlr_result is None:
        return None

    arlr_stats = localization_stats(
        gt_points,
        arlr_result['detections'],
        args.distance_thr)
    dista_result = all_results.get('DISTA-Net', {}).get(sample_idx)
    if dista_result is None:
        dista_stats = {
            'recall': 0.0,
            'precision': 0.0,
            'mean_gt_distance': 1.0,
            'det_count': 0,
        }
    else:
        dista_stats = localization_stats(
            gt_points,
            dista_result['detections'],
            args.distance_thr)
    arlr_grid_stats = reconstruction_grid_stats(
        gt_points,
        arlr_result.get('filtered_matrix', arlr_result.get('matrix')),
        metric_cfg,
        args.distance_thr)
    if dista_result is None:
        dista_grid_stats = {
            'grid_recall': 0.0,
            'grid_precision': 0.0,
            'grid_mean_gt_distance': 1.0,
            'grid_det_count': 0,
            'grid_matched': 0,
        }
    else:
        dista_grid_stats = reconstruction_grid_stats(
            gt_points,
            dista_result.get('filtered_matrix', dista_result.get('matrix')),
            metric_cfg,
            args.distance_thr)

    arlr_recall = float(arlr_stats['recall'])
    arlr_err = float(arlr_stats['mean_gt_distance'])
    arlr_precision = float(arlr_stats['precision'])
    arlr_det_count = int(arlr_stats['det_count'])
    arlr_grid_recall = float(arlr_grid_stats['grid_recall'])
    arlr_grid_precision = float(arlr_grid_stats['grid_precision'])
    arlr_grid_err = float(arlr_grid_stats['grid_mean_gt_distance'])
    arlr_grid_det_count = int(arlr_grid_stats['grid_det_count'])
    det_gap = abs(arlr_grid_det_count - gt_count)
    missing_count = max(0, gt_count - arlr_grid_det_count)
    extra_count = max(0, arlr_grid_det_count - gt_count)
    dista_recall = float(dista_stats['recall'])
    dista_err = float(dista_stats['mean_gt_distance'])
    dista_precision = float(dista_stats['precision'])
    dista_det_count = int(dista_stats['det_count'])
    dista_grid_recall = float(dista_grid_stats['grid_recall'])
    dista_grid_precision = float(dista_grid_stats['grid_precision'])
    dista_grid_err = float(dista_grid_stats['grid_mean_gt_distance'])
    dista_grid_det_count = int(dista_grid_stats['grid_det_count'])
    dista_det_gap = abs(dista_grid_det_count - gt_count)
    recall_gain = arlr_recall - dista_recall
    err_gain = dista_err - arlr_err
    grid_recall_gain = arlr_grid_recall - dista_grid_recall
    grid_err_gain = dista_grid_err - arlr_grid_err
    crowd_dist = min_pair_distance(gt_points)
    crowd_bonus = 1.0 / (crowd_dist + 0.25) if np.isfinite(crowd_dist) else 0.0
    map_diff = normalized_map_difference(
        arlr_result,
        dista_result,
        keys=('matrix', 'filtered_matrix'))
    filtered_diff = normalized_map_difference(
        arlr_result,
        dista_result,
        keys=('filtered_matrix',))

    det_gap_advantage = max(dista_det_gap - det_gap, 0)
    recon_score = (
        85.0 * arlr_grid_recall +
        28.0 * arlr_grid_precision -
        85.0 * arlr_grid_err -
        28.0 * det_gap -
        40.0 * missing_count -
        7.0 * extra_count +
        14.0 * arlr_recall +
        5.0 * arlr_precision -
        12.0 * arlr_err +
        1.5 * crowd_bonus +
        8.0 * max(grid_err_gain, 0.0) +
        7.0 * max(grid_recall_gain, 0.0) +
        4.0 * filtered_diff +
        2.0 * map_diff +
        3.0 * det_gap_advantage)
    if arlr_grid_recall >= 1.0:
        recon_score += 20.0
    if det_gap == 0:
        recon_score += 18.0
    if arlr_grid_err <= 0.05:
        recon_score += 18.0
    elif arlr_grid_err <= 0.08:
        recon_score += 11.0
    elif arlr_grid_err <= 0.10:
        recon_score += 6.0
    elif arlr_grid_err <= 0.12:
        recon_score += 3.0
    if arlr_grid_err > 0.15:
        recon_score -= 18.0
    if filtered_diff < 0.02 and map_diff < 0.02:
        recon_score -= 5.0
    if arlr_precision >= 0.8:
        recon_score += 5.0
    if arlr_grid_precision >= 0.8:
        recon_score += 6.0
    if gt_count == 4:
        recon_score -= 16.0 * det_gap
        recon_score -= 28.0 * missing_count
        if arlr_grid_err > 0.10:
            recon_score -= 8.0
    if gt_count == 5:
        if det_gap != 0:
            recon_score -= 35.0
        if arlr_grid_err > 0.12:
            recon_score -= 20.0
        if arlr_grid_recall < 1.0:
            recon_score -= 35.0

    record = {
        'sample_idx': sample_idx,
        'file_id': sample['file_id'],
        'gt_count': gt_count,
        'arlr_recall': arlr_recall,
        'arlr_err': arlr_err,
        'arlr_precision': arlr_precision,
        'arlr_filtered_det_count': arlr_grid_det_count,
        'arlr_filtered_det_count_gap': det_gap,
        'arlr_grid_recall': arlr_grid_recall,
        'arlr_grid_precision': arlr_grid_precision,
        'arlr_grid_err': arlr_grid_err,
        'arlr_grid_det_count': arlr_grid_det_count,
        'arlr_grid_det_count_gap': det_gap,
        'dista_recall': dista_recall,
        'dista_precision': dista_precision,
        'dista_err': dista_err,
        'dista_filtered_det_count': dista_grid_det_count,
        'dista_filtered_det_count_gap': dista_det_gap,
        'dista_grid_recall': dista_grid_recall,
        'dista_grid_precision': dista_grid_precision,
        'dista_grid_err': dista_grid_err,
        'dista_grid_det_count': dista_grid_det_count,
        'dista_grid_det_count_gap': dista_det_gap,
        'recall_gain': recall_gain,
        'err_gain': err_gain,
        'arlr_dista_filtered_map_diff': filtered_diff,
        'arlr_dista_map_diff': map_diff,
        'min_gt_pair_distance': crowd_dist,
        'recon_score': float(recon_score),
    }
    record.update(reconstruction_dista_distinct_metrics(record))
    return record


def reconstruction_candidate_cases(samples_by_index, all_results, args,
                                   metric_cfg):
    grouped = defaultdict(list)
    for sample_idx, sample in samples_by_index.items():
        gt_count = len(sample['gt_points'])
        if gt_count not in RECON_TARGET_ROWS:
            continue
        record = reconstruction_candidate_record(
            sample_idx,
            sample,
            all_results,
            args,
            metric_cfg)
        if record is not None:
            grouped[gt_count].append(record)
    for gt_count in grouped:
        grouped[gt_count].sort(key=lambda item: item['recon_score'],
                               reverse=True)
    return grouped


def select_reconstruction_cases_by_arlr_success(samples_by_index, all_results,
                                                 args, metric_cfg):
    grouped_candidates = reconstruction_candidate_cases(
        samples_by_index,
        all_results,
        args,
        metric_cfg)
    manual_indices = parse_recon_sample_indices_by_count(
        args.recon_sample_indices_by_count)
    records = []
    warnings_out = []
    used = set()

    def add_record(candidate, row_label, source, fallback_used, manual):
        sample = samples_by_index[candidate['sample_idx']]
        record = {
            'sample_idx': candidate['sample_idx'],
            'file_id': sample['file_id'],
            'gt_count': candidate['gt_count'],
            'row_label': row_label,
            'source': source,
            'manual': bool(manual),
            'fallback_used': bool(fallback_used),
            'selection_rule': (
                'manual_recon_sample'
                if manual else
                'strict_grid_level_pools_prioritize_dista_distinct_then_'
                'relaxed_final_recall_0.75'),
            'recon_score': candidate['recon_score'],
            'arlr_recall': candidate['arlr_recall'],
            'arlr_err': candidate['arlr_err'],
            'arlr_precision': candidate['arlr_precision'],
            'arlr_filtered_det_count': candidate['arlr_filtered_det_count'],
            'arlr_filtered_det_count_gap': (
                candidate['arlr_filtered_det_count_gap']),
            'arlr_grid_recall': candidate['arlr_grid_recall'],
            'arlr_grid_precision': candidate['arlr_grid_precision'],
            'arlr_grid_err': candidate['arlr_grid_err'],
            'arlr_grid_det_count': candidate['arlr_grid_det_count'],
            'arlr_grid_det_count_gap': candidate['arlr_grid_det_count_gap'],
            'dista_recall': candidate['dista_recall'],
            'dista_precision': candidate['dista_precision'],
            'dista_err': candidate['dista_err'],
            'dista_filtered_det_count': candidate['dista_filtered_det_count'],
            'dista_filtered_det_count_gap': (
                candidate['dista_filtered_det_count_gap']),
            'dista_grid_recall': candidate['dista_grid_recall'],
            'dista_grid_precision': candidate['dista_grid_precision'],
            'dista_grid_err': candidate['dista_grid_err'],
            'dista_grid_det_count': candidate['dista_grid_det_count'],
            'dista_grid_det_count_gap': candidate['dista_grid_det_count_gap'],
            'recall_gain': candidate['recall_gain'],
            'err_gain': candidate['err_gain'],
            'arlr_dista_filtered_map_diff': (
                candidate['arlr_dista_filtered_map_diff']),
            'arlr_dista_map_diff': candidate['arlr_dista_map_diff'],
            'grid_err_gain': candidate['grid_err_gain'],
            'grid_recall_gain': candidate['grid_recall_gain'],
            'grid_det_count_gap_advantage': (
                candidate['grid_det_count_gap_advantage']),
            'dista_distinct_score': candidate['dista_distinct_score'],
            'dista_distinct_condition_count': (
                candidate['dista_distinct_condition_count']),
            'dista_distinct_strong': candidate['dista_distinct_strong'],
            'dista_distinct_err_gain_threshold': (
                candidate['dista_distinct_err_gain_threshold']),
            'dista_distinct_filtered_diff_threshold': (
                candidate['dista_distinct_filtered_diff_threshold']),
        }
        records.append(record)
        used.add(candidate['sample_idx'])

    for gt_count, row_label in RECON_TARGET_ROWS.items():
        candidates = grouped_candidates.get(gt_count, [])
        manual_idx = manual_indices.get(gt_count)
        if manual_idx is not None:
            sample = samples_by_index.get(manual_idx)
            if sample is None:
                message = (
                    f'Reconstruction grid: manual sample {manual_idx} for '
                    f'gt_count={gt_count} was not scanned; skipping manual '
                    'override.')
                warn(message)
                warnings_out.append(message)
            elif len(sample['gt_points']) != gt_count:
                message = (
                    f'Reconstruction grid: manual sample {manual_idx} has '
                    f'gt_count={len(sample["gt_points"])}, not {gt_count}; '
                    'skipping manual override.')
                warn(message)
                warnings_out.append(message)
            else:
                manual_candidate = reconstruction_candidate_record(
                    manual_idx,
                    sample,
                    all_results,
                    args,
                    metric_cfg)
                if manual_candidate is None:
                    message = (
                        f'Reconstruction grid: manual sample {manual_idx} '
                        'has no ARLR-Net result; skipping manual override.')
                    warn(message)
                    warnings_out.append(message)
                elif manual_idx in used:
                    message = (
                        f'Reconstruction grid: manual sample {manual_idx} '
                        'was already used in another row; skipping duplicate.')
                    warn(message)
                    warnings_out.append(message)
                else:
                    add_record(
                        manual_candidate,
                        row_label,
                        'manual_recon_sample',
                        fallback_used=False,
                        manual=True)
                    continue

        if args.recon_strict_position:
            def distinct(item):
                return item['dista_distinct_strong']

            def rank_pool(pool):
                return sorted(
                    pool,
                    key=lambda item: (
                        item['dista_distinct_strong'],
                        item['dista_distinct_score'],
                        item['dista_distinct_condition_count'],
                        max(item['grid_err_gain'], 0.0),
                        item['arlr_dista_filtered_map_diff'],
                        -item['arlr_grid_err'],
                        -item['arlr_grid_det_count_gap'],
                        item['recon_score'],
                    ),
                    reverse=True)

            pools = [
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_05_dista_distinct',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.05
                        and distinct(item)
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_08_dista_distinct',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.08
                        and distinct(item)
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_10_dista_distinct',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.10
                        and distinct(item)
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_12_dista_distinct',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.12
                        and distinct(item)
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_15_dista_distinct',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.15
                        and distinct(item)
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_08',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.08
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_10',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.10
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_grid_recall_1_gap_0_grid_err_0_15',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_grid_recall'] >= 1.0
                        and item['arlr_grid_det_count_gap'] == 0
                        and item['arlr_grid_err'] <= 0.15
                        and item['sample_idx'] not in used
                    ])),
                (
                    'arlr_recall_0_75',
                    rank_pool([
                        item for item in candidates
                        if item['arlr_recall'] >= 0.75
                        and item['sample_idx'] not in used
                    ])),
                (
                    'recon_score_fallback',
                    rank_pool([
                        item for item in candidates
                        if item['sample_idx'] not in used
                    ])),
            ]
        else:
            pools = [
                (
                    'arlr_recall_1_det_count_gap_0',
                    [
                        item for item in candidates
                        if item['arlr_recall'] >= 1.0
                        and item['arlr_filtered_det_count_gap'] == 0
                        and item['sample_idx'] not in used
                    ]),
                (
                    'arlr_recall_1',
                    [
                        item for item in candidates
                        if item['arlr_recall'] >= 1.0
                        and item['sample_idx'] not in used
                    ]),
                (
                    'arlr_recall_0_75',
                    [
                        item for item in candidates
                        if item['arlr_recall'] >= 0.75
                        and item['sample_idx'] not in used
                    ]),
                (
                    'recon_score_fallback',
                    [
                        item for item in candidates
                        if item['sample_idx'] not in used
                    ]),
            ]
        chosen = None
        source = None
        fallback_used = False
        for pool_idx, (pool_name, pool) in enumerate(pools):
            if pool:
                chosen = pool[0]
                source = pool_name
                fallback_used = pool_idx > 0
                break
        if chosen is None:
            message = (
                f'Reconstruction grid: no scanned ARLR candidate with '
                f'gt_count={gt_count}; skipping {row_label}.')
            warn(message)
            warnings_out.append(message)
            continue
        add_record(
            chosen,
            row_label,
            source,
            fallback_used=fallback_used,
            manual=False)
    return records, warnings_out, grouped_candidates


def final_detection_methods_and_warnings(all_results):
    methods = []
    warnings_out = []
    for method in FINAL_DETECTION_METHOD_ORDER:
        if method == 'GT':
            methods.append(method)
        elif method in all_results:
            methods.append(method)
        else:
            message = (
                f'Final detection grid: {method} has no successful inference '
                'result and is skipped.')
            warn(message)
            warnings_out.append(message)
    return methods, warnings_out


def final_detection_candidate_record(sample_idx, sample, all_results, args):
    gt_points = sample['gt_points']
    gt_count = len(gt_points)
    arlr_result = all_results.get('ARLR-Net', {}).get(sample_idx)
    if arlr_result is None:
        return None
    arlr_stats = localization_stats(
        gt_points,
        arlr_result['detections'],
        args.distance_thr)
    dista_result = all_results.get('DISTA-Net', {}).get(sample_idx)
    if dista_result is None:
        dista_stats = {
            'recall': 0.0,
            'precision': 0.0,
            'mean_gt_distance': 1.0,
            'det_count': 0,
        }
    else:
        dista_stats = localization_stats(
            gt_points,
            dista_result['detections'],
            args.distance_thr)

    arlr_recall = float(arlr_stats['recall'])
    arlr_err = float(arlr_stats['mean_gt_distance'])
    arlr_precision = float(arlr_stats['precision'])
    dista_recall = float(dista_stats['recall'])
    dista_err = float(dista_stats['mean_gt_distance'])
    recall_gain = arlr_recall - dista_recall
    err_gain = dista_err - arlr_err
    crowd_dist = min_pair_distance(gt_points)
    crowd_bonus = 1.0 / (crowd_dist + 0.25) if np.isfinite(crowd_dist) else 0.0
    det_gap = abs(int(arlr_stats['det_count']) - gt_count)
    distinct_bonus = (
        8.0 * max(err_gain, 0.0) +
        7.0 * max(recall_gain, 0.0))
    score = (
        55.0 * arlr_recall -
        65.0 * arlr_err +
        9.0 * arlr_precision -
        1.5 * det_gap +
        1.2 * crowd_bonus +
        distinct_bonus)
    if arlr_recall >= 1.0:
        score += 10.0
    if arlr_err <= 0.08:
        score += 8.0
    elif arlr_err <= 0.10:
        score += 4.0
    elif arlr_err > 0.15:
        score -= 8.0
    if err_gain > 0.0 or recall_gain > 0.0:
        score += 3.0

    return {
        'sample_idx': sample_idx,
        'file_id': sample['file_id'],
        'gt_count': gt_count,
        'arlr_recall': arlr_recall,
        'arlr_err': arlr_err,
        'arlr_precision': arlr_precision,
        'arlr_det_count': int(arlr_stats['det_count']),
        'dista_recall': dista_recall,
        'dista_err': dista_err,
        'recall_gain': recall_gain,
        'err_gain': err_gain,
        'min_gt_pair_distance': crowd_dist,
        'final_detection_score': float(score),
    }


def select_final_detection_cases(samples_by_index, all_results, args):
    grouped = defaultdict(list)
    for sample_idx, sample in samples_by_index.items():
        gt_count = len(sample['gt_points'])
        if gt_count not in RECON_TARGET_ROWS:
            continue
        record = final_detection_candidate_record(
            sample_idx,
            sample,
            all_results,
            args)
        if record is not None:
            grouped[gt_count].append(record)
    for gt_count in grouped:
        grouped[gt_count].sort(
            key=lambda item: item['final_detection_score'],
            reverse=True)

    records = []
    warnings_out = []
    used = set()
    for gt_count, row_label in RECON_TARGET_ROWS.items():
        candidates = grouped.get(gt_count, [])
        pools = [
            (
                'arlr_recall_1_err_0_08_distinct',
                [
                    item for item in candidates
                    if item['arlr_recall'] >= 1.0
                    and item['arlr_err'] <= 0.08
                    and (item['err_gain'] > 0.0 or item['recall_gain'] > 0.0)
                    and item['sample_idx'] not in used
                ]),
            (
                'arlr_recall_1_err_0_08',
                [
                    item for item in candidates
                    if item['arlr_recall'] >= 1.0
                    and item['arlr_err'] <= 0.08
                    and item['sample_idx'] not in used
                ]),
            (
                'arlr_recall_1_err_0_10_distinct',
                [
                    item for item in candidates
                    if item['arlr_recall'] >= 1.0
                    and item['arlr_err'] <= 0.10
                    and (item['err_gain'] > 0.0 or item['recall_gain'] > 0.0)
                    and item['sample_idx'] not in used
                ]),
            (
                'arlr_recall_1_err_0_15',
                [
                    item for item in candidates
                    if item['arlr_recall'] >= 1.0
                    and item['arlr_err'] <= 0.15
                    and item['sample_idx'] not in used
                ]),
            (
                'arlr_recall_0_75',
                [
                    item for item in candidates
                    if item['arlr_recall'] >= 0.75
                    and item['sample_idx'] not in used
                ]),
            (
                'final_detection_score_fallback',
                [
                    item for item in candidates
                    if item['sample_idx'] not in used
                ]),
        ]
        chosen = None
        source = None
        fallback_used = False
        for pool_idx, (pool_name, pool) in enumerate(pools):
            if pool:
                chosen = pool[0]
                source = pool_name
                fallback_used = pool_idx > 0
                break
        if chosen is None:
            message = (
                f'Final detection grid: no scanned ARLR candidate with '
                f'gt_count={gt_count}; skipping {row_label}.')
            warn(message)
            warnings_out.append(message)
            continue
        record = dict(chosen)
        record.update({
            'row_label': row_label,
            'source': source,
            'fallback_used': bool(fallback_used),
        })
        records.append(record)
        used.add(chosen['sample_idx'])
    return records, warnings_out


def draw_detection_points(ax, gt_points, detections, matches, show_unmatched):
    gt_arr = np.asarray([[point[0], point[1]] for point in gt_points],
                        dtype=np.float64)
    if gt_arr.size:
        ax.scatter(
            gt_arr[:, 1],
            gt_arr[:, 0],
            s=34,
            marker='x',
            linewidths=1.45,
            c=GT_COLOR,
            zorder=5)
    detections = detections or []
    matched_det_indices = {det_idx for _, det_idx, _ in matches}
    if show_unmatched:
        unmatched = [
            det for idx, det in enumerate(detections)
            if idx not in matched_det_indices
        ][:80]
        if unmatched:
            ax.scatter(
                [det['col'] for det in unmatched],
                [det['row'] for det in unmatched],
                s=12,
                marker='o',
                facecolors='none',
                edgecolors=PRED_COLOR,
                linewidths=0.55,
                alpha=0.16,
                zorder=4)
    matched = [
        detections[det_idx]
        for _, det_idx, _ in matches
        if det_idx < len(detections)
    ]
    if matched:
        ax.scatter(
            [det['col'] for det in matched],
            [det['row'] for det in matched],
            s=58,
            marker='o',
            facecolors='none',
            edgecolors=PRED_COLOR,
            linewidths=1.9,
            zorder=6)


def draw_final_detection_comparison_grid(samples_by_index, all_results, out_dir,
                                         args):
    methods, method_warnings = final_detection_methods_and_warnings(all_results)
    records, case_warnings = select_final_detection_cases(
        samples_by_index,
        all_results,
        args)
    summary = {
        'samples': records,
        'methods': methods,
        'warnings': method_warnings + case_warnings,
        'selection_rule': (
            'For each gt_count=3/4/5, select from all scanned samples using '
            'ARLR final localization quality. Prefer recall=1.0, '
            'arlr_err<=0.08 or <=0.10, and ARLR advantage over DISTA.'),
        'display': (
            'Filtered final intensity map background with GT x markers and '
            'matched final prediction circles. ARLR detections use x_final '
            'plus subpixel_offset when enabled by the evaluator config.'),
    }
    if not records:
        warn('Final detection grid: no suitable rows selected; skipping figure.')
        return summary

    rows = len(records)
    cols = len(methods)
    fig_width = max(8.0, cols * 1.25)
    fig_height = max(3.2, rows * 1.55)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False)
    fig.patch.set_facecolor('white')

    for row_idx, record in enumerate(records):
        sample_idx = record['sample_idx']
        sample = samples_by_index[sample_idx]
        gt_points = sample['gt_points']
        crop_bounds = reconstruction_crop_bounds(
            sample,
            sample['gt_matrix'].shape)
        if crop_bounds is not None:
            record['crop_bounds'] = {
                'row_min': crop_bounds[0],
                'row_max': crop_bounds[1],
                'col_min': crop_bounds[2],
                'col_max': crop_bounds[3],
            }
        vmax = row_vmax_for_reconstruction(
            sample_idx,
            samples_by_index,
            all_results,
            methods,
            crop_bounds,
            'filtered',
            summary['warnings'],
            set())
        norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            matrix = matrix_for_reconstruction_panel(
                method,
                sample_idx,
                samples_by_index,
                all_results,
                'filtered',
                summary['warnings'],
                set())
            if matrix is None:
                ax.axis('off')
                continue
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            display_matrix = crop_matrix_for_display(matrix, crop_bounds)
            extent = matrix_extent(matrix.shape, crop_bounds)
            ax.imshow(
                black_floor_matrix(display_matrix),
                cmap=black_floor_cmap(args.background_cmap),
                origin='lower',
                interpolation='nearest',
                norm=norm,
                extent=extent,
                alpha=args.background_alpha)
            if method == 'ARLR-Net':
                spine_width = 1.9
            elif method == 'DISTA-Net':
                spine_width = 1.4
            else:
                spine_width = 0.8
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#FF3B30')
                spine.set_linewidth(spine_width)
            ax.set_xticks([])
            ax.set_yticks([])
            if method == 'GT':
                draw_detection_points(ax, gt_points, [], [], False)
                label = method
                label_color = GT_COLOR
            else:
                result = all_results.get(method, {}).get(sample_idx)
                detections = result.get('detections', []) if result else []
                matches = greedy_matches(gt_points, detections, args.distance_thr)
                stats = localization_stats(
                    gt_points,
                    detections,
                    args.distance_thr)
                draw_detection_points(
                    ax,
                    gt_points,
                    detections,
                    matches,
                    args.show_unmatched_faint)
                label = (
                    f'{method}\n'
                    f'R={stats["recall"]:.2f} err={stats["mean_gt_distance"]:.3f}')
                label_color = '#00E5FF' if method == 'ARLR-Net' else 'white'
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            label_box = dict(
                facecolor='black',
                edgecolor='none',
                alpha=0.68,
                pad=1.05)
            if method == 'ARLR-Net':
                label_box.update(edgecolor='#00E5FF', linewidth=0.45, alpha=0.78)
            label_pos = reconstruction_label_position(display_matrix)
            ax.text(
                label_pos['x'],
                label_pos['y'],
                label,
                transform=ax.transAxes,
                ha=label_pos['ha'],
                va=label_pos['va'],
                fontsize=6.2,
                fontweight=(
                    'bold'
                    if method in ('GT', 'ARLR-Net')
                    else 'normal'),
                color=label_color,
                linespacing=1.05,
                bbox=label_box)
            if col_idx == 0:
                ax.text(
                    -0.10,
                    0.5,
                    record['row_label'],
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=8.5,
                    color='black')

    fig.tight_layout(pad=0.35, w_pad=0.25, h_pad=0.28)
    save_figure(fig, out_dir, 'final_detection_comparison_grid')
    return summary


def draw_reconstruction_aligned_localization_comparison(
        samples_by_index, all_results, reconstruction_summary, out_dir, args):
    if not args.save_reconstruction_aligned_localization:
        return None
    records = list(reconstruction_summary.get('samples', []))
    summary = {
        'uses_same_samples_as_reconstruction_comparison_grid': True,
        'samples': [],
        'warnings': [],
        'methods': ['GT', 'DISTA-Net', 'ARLR-Net'],
    }
    if not records:
        warn('Reconstruction-aligned localization comparison: no '
             'reconstruction samples available; skipping figure.')
        summary['warnings'].append('No reconstruction samples available.')
        return summary
    for method in ('DISTA-Net', 'ARLR-Net'):
        if method not in all_results:
            message = (
                f'Reconstruction-aligned localization comparison: {method} '
                'has no successful inference result; skipping figure.')
            warn(message)
            summary['warnings'].append(message)
            return summary

    rows = len(records)
    methods = ['GT', 'DISTA-Net', 'ARLR-Net']
    fig, axes = plt.subplots(
        rows,
        len(methods),
        figsize=(9.2, max(2.75, rows * 2.55)),
        squeeze=False)
    fig.patch.set_facecolor('white')
    fig.suptitle('Reconstruction-aligned Localization Comparison',
                 fontsize=14,
                 y=1.01)

    for row_idx, record in enumerate(records):
        sample_idx = record['sample_idx']
        sample = samples_by_index.get(sample_idx)
        if sample is None:
            message = (
                f'Reconstruction-aligned localization comparison: sample '
                f'{sample_idx} is unavailable; row skipped.')
            warn(message)
            summary['warnings'].append(message)
            for ax in axes[row_idx]:
                ax.axis('off')
            continue

        gt_points = sample['gt_points']
        result_by_method = {
            method: all_results[method].get(sample_idx)
            for method in ('DISTA-Net', 'ARLR-Net')
        }
        zoom = zoom_for_case(
            gt_points,
            result_by_method,
            ('DISTA-Net', 'ARLR-Net'),
            args.distance_thr,
            pad=0.95)
        limits = intensity_limits_for_case(
            [result_by_method['DISTA-Net'], result_by_method['ARLR-Net']],
            sample['gt_matrix'])
        row_summary = {
            'sample_idx': sample_idx,
            'gt_count': record.get('gt_count', len(gt_points)),
            'row_label': record.get('row_label'),
            'file_id': record.get('file_id', sample['file_id']),
            'manual': bool(record.get('manual', False)),
            'source': record.get('source'),
            'zoom': {
                'xmin': zoom[0],
                'xmax': zoom[1],
                'ymin': zoom[2],
                'ymax': zoom[3],
            },
        }

        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            title = method if row_idx == 0 else ''
            if method == 'GT':
                matrix = None if args.hide_intensity_background else sample['gt_matrix']
                plot_points(
                    ax,
                    gt_points,
                    title=title,
                    matrix=matrix,
                    plot_topk=args.plot_topk_dets,
                    zoom=zoom,
                    hide_ticks=True,
                    intensity_limits=limits,
                    gt_size=76,
                    pred_size_range=(56, 96),
                    line_width=1.05,
                    matched_only=args.plot_matched_only,
                    background_cmap=args.background_cmap,
                    background_alpha=args.background_alpha,
                    show_unmatched_faint=args.show_unmatched_faint,
                    show_error_lines=(
                        args.show_error_lines
                        and not args.clean_localization_style))
            else:
                result = result_by_method[method]
                detections = result.get('detections', []) if result else []
                matches = greedy_matches(gt_points, detections, args.distance_thr)
                stats = localization_stats(gt_points, detections,
                                           args.distance_thr)
                subtitle = (
                    f'R={stats["recall"]:.2f}, '
                    f'err={stats["mean_gt_distance"]:.3f}')
                if title:
                    title = f'{title}\n{subtitle}'
                else:
                    title = subtitle
                plot_points(
                    ax,
                    gt_points,
                    detections,
                    matches,
                    title=title,
                    matrix=(None if args.hide_intensity_background
                            else result['filtered_matrix']),
                    plot_topk=args.plot_topk_dets,
                    zoom=zoom,
                    hide_ticks=True,
                    intensity_limits=limits,
                    gt_size=76,
                    pred_size_range=(56, 96),
                    line_width=1.05,
                    matched_only=args.plot_matched_only,
                    background_cmap=args.background_cmap,
                    background_alpha=args.background_alpha,
                    show_unmatched_faint=args.show_unmatched_faint,
                    show_error_lines=(
                        args.show_error_lines
                        and not args.clean_localization_style))
                if method == 'ARLR-Net':
                    ax.title.set_fontweight('bold')
                    ax.title.set_color('#1F2937')
                row_summary[method] = {
                    'recall': float(stats['recall']),
                    'err': float(stats['mean_gt_distance']),
                    'precision': float(stats['precision']),
                    'det_count': int(stats['det_count']),
                }

            if col_idx == 0:
                ax.text(
                    -0.10,
                    0.5,
                    record.get('row_label', ''),
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=8.5,
                    color='black')
        summary['samples'].append(row_summary)

    fig.tight_layout(pad=0.68, rect=(0.0, 0.08, 1.0, 0.96))
    add_localization_legend(
        fig,
        y=0.012,
        show_error_lines=(
            args.show_error_lines and not args.clean_localization_style))
    save_figure(fig, out_dir, 'reconstruction_aligned_localization_comparison')
    return summary


def draw_paper_representative_localization_grid(
        samples_by_index, all_results, reconstruction_summary, out_dir, args):
    records = list(reconstruction_summary.get('samples', []))
    requested_methods = [
        'GT', 'ARLR-Net', 'DISTA-Net', 'ISTA', 'SRFBN', 'GGLNet'
    ]
    summary = {
        'uses_same_samples_as_reconstruction_comparison_grid': True,
        'requested_methods': requested_methods,
        'methods': [],
        'sample_indices': [],
        'samples': [],
        'warnings': [],
        'display': (
            'Filtered final intensity map background with yellow GT x markers '
            'and cyan hollow circles for matched predictions only. Unmatched '
            'predictions are faintly shown only when show_unmatched_faint=True. '
            'The bottom marker legend is intentionally omitted.'),
        'normalization': {
            'mode': 'per-row shared PowerNorm over the shared zoom region',
            'gamma': 0.62,
            'vmin': 0.0,
            'vmax': '99.2 percentile of positive pixels in the row zoom',
        },
        'legend': 'omitted',
    }
    if not records:
        message = (
            'Paper representative localization grid: no reconstruction '
            'samples available; skipping figure.')
        warn(message)
        summary['warnings'].append(message)
        return summary

    methods = []
    for method in requested_methods:
        if method == 'GT':
            methods.append(method)
            continue
        if method in all_results:
            missing_samples = [
                record.get('sample_idx')
                for record in records
                if record.get('sample_idx') not in all_results[method]
            ]
            if missing_samples:
                message = (
                    f'Paper representative localization grid: {method} is in '
                    'all_results but lacks result entries for '
                    'reconstruction-grid '
                    f'samples {missing_samples}; skipping the column.')
                warn(message)
                summary['warnings'].append(message)
            else:
                methods.append(method)
        else:
            message = (
                f'Paper representative localization grid: {method} is not in '
                'all_results because no successful inference result was '
                'produced; skipping the column.')
            warn(message)
            summary['warnings'].append(message)
    summary['methods'] = methods
    summary['sample_indices'] = [record.get('sample_idx') for record in records]

    rows = len(records)
    cols = len(methods)
    fig_width = max(9.2, cols * 1.62)
    fig_height = max(4.8, rows * 1.95)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False)
    fig.patch.set_facecolor('white')

    def zoomed_intensity_limits(results, gt_matrix, zoom):
        values = []
        if gt_matrix is not None:
            gt_crop = matrix_crop_from_zoom(gt_matrix, zoom)
            positives = gt_crop[gt_crop > 0]
            if positives.size:
                values.append(positives)
        for result in results:
            if result is None:
                continue
            matrix = result.get('filtered_matrix')
            if matrix is None:
                continue
            crop = matrix_crop_from_zoom(matrix, zoom)
            positives = crop[crop > 0]
            if positives.size:
                values.append(positives)
        if not values:
            return None
        merged = np.concatenate(values)
        vmax = float(np.percentile(merged, 99.2))
        if vmax <= 0:
            vmax = float(merged.max())
        return 0.0, max(vmax, 1e-6)

    for row_idx, record in enumerate(records):
        sample_idx = record['sample_idx']
        sample = samples_by_index.get(sample_idx)
        if sample is None:
            message = (
                f'Paper representative localization grid: sample {sample_idx} '
                'is unavailable; row skipped.')
            warn(message)
            summary['warnings'].append(message)
            for ax in axes[row_idx]:
                ax.axis('off')
            continue

        gt_points = sample['gt_points']
        result_by_method = {
            method: all_results[method].get(sample_idx)
            for method in methods
            if method != 'GT'
        }
        zoom = zoom_for_case(
            gt_points,
            result_by_method,
            result_by_method.keys(),
            args.distance_thr,
            pad=0.95)
        intensity_limits = zoomed_intensity_limits(
            list(result_by_method.values()),
            sample['gt_matrix'],
            zoom)
        row_summary = {
            'sample_idx': sample_idx,
            'file_id': record.get('file_id', sample['file_id']),
            'gt_count': record.get('gt_count', len(gt_points)),
            'row_label': record.get('row_label'),
            'source': record.get('source'),
            'manual': bool(record.get('manual', False)),
            'method_metrics': {},
            'zoom': {
                'xmin': zoom[0],
                'xmax': zoom[1],
                'ymin': zoom[2],
                'ymax': zoom[3],
            },
        }

        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            if method == 'GT':
                plot_points(
                    ax,
                    gt_points,
                    title='GT',
                    matrix=sample['gt_matrix'],
                    plot_topk=args.plot_topk_dets,
                    zoom=zoom,
                    hide_ticks=True,
                    intensity_limits=intensity_limits,
                    gt_size=76,
                    pred_size_range=(56, 96),
                    matched_only=True,
                    background_cmap=args.background_cmap,
                    background_alpha=args.background_alpha,
                    show_unmatched_faint=False,
                    show_error_lines=False)
                ax.set_title('GT', fontsize=10.5, pad=3.0)
            else:
                result = result_by_method.get(method)
                detections = result.get('detections', []) if result else []
                matches = greedy_matches(gt_points, detections, args.distance_thr)
                stats = localization_stats(
                    gt_points,
                    detections,
                    args.distance_thr)
                plot_points(
                    ax,
                    gt_points,
                    detections,
                    matches,
                    title=(
                        f'{method}\nR={stats["recall"]:.2f}, '
                        f'err={stats["mean_gt_distance"]:.4f}'),
                    matrix=result.get('filtered_matrix') if result else None,
                    plot_topk=args.plot_topk_dets,
                    zoom=zoom,
                    hide_ticks=True,
                    intensity_limits=intensity_limits,
                    gt_size=76,
                    pred_size_range=(56, 96),
                    matched_only=True,
                    background_cmap=args.background_cmap,
                    background_alpha=args.background_alpha,
                    show_unmatched_faint=args.show_unmatched_faint,
                    show_error_lines=False)
                ax.set_title(
                    f'{method}\nR={stats["recall"]:.2f}, '
                    f'err={stats["mean_gt_distance"]:.4f}',
                    fontsize=9.2,
                    pad=3.0,
                    linespacing=0.98)
                if method == 'ARLR-Net':
                    ax.title.set_fontweight('bold')
                    ax.title.set_color('black')
                method_summary = {
                    'recall': float(stats['recall']),
                    'err': float(stats['mean_gt_distance']),
                    'precision': float(stats['precision']),
                    'det_count': int(stats['det_count']),
                }
                row_summary['method_metrics'][method] = method_summary
                row_summary[method] = method_summary

            if method == 'ARLR-Net':
                spine_width = 1.8
            elif method == 'DISTA-Net':
                spine_width = 1.35
            else:
                spine_width = 0.8
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#FF3B30')
                spine.set_linewidth(spine_width)
            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    record.get('row_label', ''),
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=13.0,
                    color='black')
        summary['samples'].append(row_summary)

    fig.tight_layout(pad=0.70, w_pad=0.35, h_pad=0.45)
    save_figure(fig, out_dir, 'paper_representative_localization_grid')
    return summary


def draw_reconstruction_comparison_grid(samples_by_index, selected,
                                        all_results, out_dir, args,
                                        metric_cfg,
                                        brightness_threshold=None):
    recon_methods = list(RECON_METHOD_ORDER)
    methods, method_warnings = reconstruction_methods_and_warnings(
        all_results,
        recon_methods)
    records, case_warnings, grouped_candidates = (
        select_reconstruction_cases_by_arlr_success(
            samples_by_index,
            all_results,
            args,
            metric_cfg))
    candidate_cases_top30 = []
    for gt_count in RECON_TARGET_ROWS:
        for item in grouped_candidates.get(gt_count, [])[:30]:
            candidate_cases_top30.append(dict(item))
    summary = {
        'samples': records,
        'methods': methods,
        'warnings': method_warnings + case_warnings,
        'selection_rule': (
            'For each gt_count=3/4/5, select from all scanned samples with '
            'ARLR-Net results. In strict position mode, first require '
            'ARLR grid-level recall=1.0, grid det_count gap 0, and low '
            'grid-level error from filtered_matrix without subpixel offset, '
            'while prioritizing candidates distinct from DISTA-Net by '
            'grid-level error gain, filtered-map difference, recall gain, '
            'or det_count-gap advantage. This distinct priority is strongest '
            'for gt_count=3 and gt_count=4. Then relax through non-distinct '
            'low-error grid pools, final recall>=0.75, and finally '
            'recon_score fallback.'),
        'dista_distinct_thresholds': {
            str(gt_count): {
                'grid_err_gain': RECON_DISTINCT_ERR_GAIN_THRESHOLDS[gt_count],
                'filtered_map_diff': (
                    RECON_DISTINCT_FILTERED_DIFF_THRESHOLDS[gt_count]),
            }
            for gt_count in RECON_TARGET_ROWS
        },
        'recon_strict_position': bool(args.recon_strict_position),
        'candidate_cases_top30': candidate_cases_top30,
        'candidate_counts': {
            str(gt_count): len(grouped_candidates.get(gt_count, []))
            for gt_count in RECON_TARGET_ROWS
        },
        'display_map_type': args.recon_map_type,
        'brightness_threshold': brightness_threshold,
        'filtered_map_note': (
            'Filtered reconstruction maps follow the same brightness_threshold '
            'rule used by localization evaluation.'),
        'normalization': {
            'mode': 'per-row shared PowerNorm',
            'gamma': 0.55,
            'vmin': 0.0,
            'vmax': '99.5 percentile of positive pixels in the row',
        },
        'display': {
            'background': 'white figure, black reconstruction panels',
            'crop': 'per-row shared crop around GT targets when useful',
            'label_placement': 'automatic low-energy corner per panel',
        },
    }
    if not records:
        warn('Reconstruction grid: no suitable rows selected; skipping figure.')
        return summary

    rows = len(records)
    cols = len(methods)
    fig_width = max(8.5, cols * 1.18)
    fig_height = max(3.2, rows * 1.55)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False)
    fig.patch.set_facecolor('white')
    fallback_warning_keys = set()

    for row_idx, record in enumerate(records):
        sample_idx = record['sample_idx']
        sample = samples_by_index[sample_idx]
        crop_bounds = reconstruction_crop_bounds(
            sample,
            sample['gt_matrix'].shape)
        if crop_bounds is not None:
            record['crop_bounds'] = {
                'row_min': crop_bounds[0],
                'row_max': crop_bounds[1],
                'col_min': crop_bounds[2],
                'col_max': crop_bounds[3],
            }
        vmax = row_vmax_for_reconstruction(
            sample_idx,
            samples_by_index,
            all_results,
            methods,
            crop_bounds,
            args.recon_map_type,
            summary['warnings'],
            fallback_warning_keys)
        norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            matrix = matrix_for_reconstruction_panel(
                method,
                sample_idx,
                samples_by_index,
                all_results,
                args.recon_map_type,
                summary['warnings'],
                fallback_warning_keys)
            if matrix is None:
                ax.set_facecolor('white')
                ax.axis('off')
                message = (
                    f'Reconstruction grid: missing {method} result for '
                    f'sample {sample_idx}; cell left blank.')
                warn(message)
                summary['warnings'].append(message)
                continue
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            display_matrix = crop_matrix_for_display(matrix, crop_bounds)
            ax.imshow(
                display_matrix,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            if method == 'ARLR-Net':
                spine_width = 1.8
            elif method == 'DISTA-Net':
                spine_width = 1.35
            else:
                spine_width = 0.8
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#FF3B30')
                spine.set_linewidth(spine_width)
            if method == 'GT':
                label_color = '#F6C343'
            elif method == 'ARLR-Net':
                label_color = '#00E5FF'
            else:
                label_color = 'white'
            label_box = dict(
                facecolor='black',
                edgecolor='none',
                alpha=0.62,
                pad=1.2)
            if method == 'ARLR-Net':
                label_box.update(
                    edgecolor='#00E5FF',
                    linewidth=0.45,
                    alpha=0.74)
            label_pos = reconstruction_label_position(display_matrix)
            ax.text(
                label_pos['x'],
                label_pos['y'],
                method,
                transform=ax.transAxes,
                ha=label_pos['ha'],
                va=label_pos['va'],
                fontsize=10.0,
                fontweight=(
                    'bold'
                    if method in ('GT', 'ARLR-Net')
                    else 'normal'),
                color=label_color,
                bbox=label_box)
            if col_idx == 0:
                ax.text(
                    -0.12,
                    0.5,
                    record['row_label'],
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=10.5,
                    color='black')

    fig.tight_layout(pad=0.35, w_pad=0.28, h_pad=0.0)
    fig.subplots_adjust(hspace=-0.48)
    save_figure(fig, out_dir, 'reconstruction_comparison_grid')
    return summary


def draw_reconstruction_false_alarm_grid(
        samples_by_index, all_results, reconstruction_summary, out_dir, args):
    records = list(reconstruction_summary.get('samples', []))
    methods = list(reconstruction_summary.get('methods', []))
    summary = {
        'uses_same_samples_as_reconstruction_comparison_grid': True,
        'methods': methods,
        'samples': [],
        'warnings': [],
        'display_map_type': args.recon_map_type,
        'display': {
            'background': (
                'same reconstruction map, crop, normalization, and labels as '
                'reconstruction_comparison_grid'),
            'overlay': (
                'small red hollow circles mark unmatched predictions only; '
                'GT, matched predictions, and GT markers are not overlaid'),
        },
    }
    if not records:
        warn('Reconstruction false-alarm grid: no reconstruction samples; '
             'skipping figure.')
        return summary
    if not methods:
        warn('Reconstruction false-alarm grid: no reconstruction methods; '
             'skipping figure.')
        return summary

    rows = len(records)
    cols = len(methods)
    fig_width = max(8.5, cols * 1.18)
    fig_height = max(3.2, rows * 1.55)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_width, fig_height),
        squeeze=False)
    fig.patch.set_facecolor('white')
    fallback_warning_keys = set()

    for row_idx, record in enumerate(records):
        sample_idx = record['sample_idx']
        sample = samples_by_index.get(sample_idx)
        if sample is None:
            message = (
                f'Reconstruction false-alarm grid: sample {sample_idx} is '
                'unavailable; row skipped.')
            warn(message)
            summary['warnings'].append(message)
            for ax in axes[row_idx]:
                ax.axis('off')
            continue

        gt_points = sample['gt_points']
        crop_bounds = reconstruction_crop_bounds(
            sample,
            sample['gt_matrix'].shape)
        vmax = row_vmax_for_reconstruction(
            sample_idx,
            samples_by_index,
            all_results,
            methods,
            crop_bounds,
            args.recon_map_type,
            summary['warnings'],
            fallback_warning_keys)
        norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
        row_summary = {
            'sample_idx': sample_idx,
            'file_id': record.get('file_id', sample['file_id']),
            'gt_count': record.get('gt_count', len(gt_points)),
            'row_label': record.get('row_label'),
            'method_unmatched_counts': {},
        }

        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            matrix = matrix_for_reconstruction_panel(
                method,
                sample_idx,
                samples_by_index,
                all_results,
                args.recon_map_type,
                summary['warnings'],
                fallback_warning_keys)
            if matrix is None:
                ax.set_facecolor('white')
                ax.axis('off')
                message = (
                    f'Reconstruction false-alarm grid: missing {method} '
                    f'result for sample {sample_idx}; cell left blank.')
                warn(message)
                summary['warnings'].append(message)
                row_summary['method_unmatched_counts'][method] = None
                continue
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            display_matrix = crop_matrix_for_display(matrix, crop_bounds)
            ax.imshow(
                display_matrix,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

            unmatched = []
            if method != 'GT':
                result = all_results.get(method, {}).get(sample_idx)
                detections = result.get('detections', []) if result else []
                matches = greedy_matches(
                    gt_points,
                    detections,
                    args.distance_thr)
                matched_indices = {det_idx for _, det_idx, _ in matches}
                unmatched = [
                    det for det_idx, det in enumerate(detections)
                    if det_idx not in matched_indices
                ]
                row_summary['method_unmatched_counts'][method] = len(unmatched)
            else:
                row_summary['method_unmatched_counts'][method] = 0

            if unmatched:
                if crop_bounds is None:
                    row_offset = 0
                    col_offset = 0
                    row_max, col_max = display_matrix.shape
                else:
                    row_offset = crop_bounds[0]
                    col_offset = crop_bounds[2]
                    row_max = crop_bounds[1] - crop_bounds[0]
                    col_max = crop_bounds[3] - crop_bounds[2]
                circle_rows = []
                circle_cols = []
                for det in unmatched:
                    det_row = det.get('grid_row')
                    det_col = det.get('grid_col')
                    if det_row is None or det_col is None:
                        continue
                    plot_row = int(det_row) - row_offset
                    plot_col = int(det_col) - col_offset
                    if 0 <= plot_row < row_max and 0 <= plot_col < col_max:
                        circle_rows.append(plot_row)
                        circle_cols.append(plot_col)
                if circle_rows:
                    ax.scatter(
                        circle_cols,
                        circle_rows,
                        s=24,
                        marker='o',
                        facecolors='none',
                        edgecolors='#FF2D20',
                        linewidths=0.9,
                        zorder=7)

            if method == 'ARLR-Net':
                spine_width = 1.8
            else:
                spine_width = 0.8
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#FF3B30')
                spine.set_linewidth(spine_width)
            if method == 'GT':
                label_color = '#F6C343'
            elif method == 'ARLR-Net':
                label_color = '#00E5FF'
            else:
                label_color = 'white'
            label_box = dict(
                facecolor='black',
                edgecolor='none',
                alpha=0.62,
                pad=1.2)
            if method == 'ARLR-Net':
                label_box.update(
                    edgecolor='#00E5FF',
                    linewidth=0.45,
                    alpha=0.74)
            label_pos = reconstruction_label_position(display_matrix)
            ax.text(
                label_pos['x'],
                label_pos['y'],
                method,
                transform=ax.transAxes,
                ha=label_pos['ha'],
                va=label_pos['va'],
                fontsize=7.0,
                fontweight=(
                    'bold'
                    if method in ('GT', 'ARLR-Net')
                    else 'normal'),
                color=label_color,
                bbox=label_box)
            if col_idx == 0:
                ax.text(
                    -0.10,
                    0.5,
                    record['row_label'],
                    transform=ax.transAxes,
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=10.5,
                    color='black')
        summary['samples'].append(row_summary)

    fig.tight_layout(pad=0.35, w_pad=0.28, h_pad=0.0)
    fig.subplots_adjust(hspace=-0.48)
    save_figure(fig, out_dir, 'reconstruction_false_alarm_grid')
    return summary


def draw_reconstruction_candidate_preview_5targets(
        samples_by_index, all_results, out_dir, args, reconstruction_summary):
    if not args.save_recon_candidate_preview:
        return None
    candidates = [
        item for item in reconstruction_summary.get('candidate_cases_top30', [])
        if int(item.get('gt_count', -1)) == 5
    ][:max(1, int(args.recon_candidate_preview_top))]
    if not candidates:
        warn('Five Targets reconstruction candidate preview: no candidates.')
        return None

    methods = ['GT', 'ARLR-Net', 'DISTA-Net']
    rows = len(candidates)
    cols = len(methods)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 1.85, max(2.5, rows * 1.45)),
        squeeze=False)
    fig.patch.set_facecolor('white')
    summary = {
        'num_candidates': rows,
        'sample_indices': [item['sample_idx'] for item in candidates],
    }
    fallback_warning_keys = set()
    for row_idx, candidate in enumerate(candidates):
        sample_idx = candidate['sample_idx']
        sample = samples_by_index[sample_idx]
        crop_bounds = reconstruction_crop_bounds(
            sample,
            sample['gt_matrix'].shape,
            pad=5,
            min_size=23)
        vmax = row_vmax_for_reconstruction(
            sample_idx,
            samples_by_index,
            all_results,
            methods,
            crop_bounds,
            'filtered',
            None,
            fallback_warning_keys)
        norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('black')
            matrix = matrix_for_reconstruction_panel(
                method,
                sample_idx,
                samples_by_index,
                all_results,
                'filtered',
                None,
                fallback_warning_keys)
            if matrix is None:
                ax.axis('off')
                continue
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
            display_matrix = crop_matrix_for_display(matrix, crop_bounds)
            ax.imshow(
                display_matrix,
                cmap='gray',
                origin='lower',
                interpolation='nearest',
                norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('#FF3B30')
                spine.set_linewidth(1.4 if method == 'ARLR-Net' else 0.8)
            if col_idx == 0:
                label = (
                    f'#{sample_idx}\n'
                    f'gerr={candidate["arlr_grid_err"]:.3f}\n'
                    f'gR={candidate["arlr_grid_recall"]:.2f} '
                    f'diff={candidate["arlr_dista_filtered_map_diff"]:.3f}')
            else:
                label = method
            label_pos = reconstruction_label_position(display_matrix)
            ax.text(
                label_pos['x'],
                label_pos['y'],
                label,
                transform=ax.transAxes,
                ha=label_pos['ha'],
                va=label_pos['va'],
                fontsize=5.8,
                color=('#00E5FF' if method == 'ARLR-Net'
                       else (GT_COLOR if method == 'GT' else 'white')),
                fontweight='bold' if method in ('GT', 'ARLR-Net') else 'normal',
                linespacing=1.0,
                bbox=dict(
                    facecolor='black',
                    edgecolor='none',
                    alpha=0.68,
                    pad=0.95))
            if row_idx == 0:
                ax.set_title(method, fontsize=8, color='black')
    fig.tight_layout(pad=0.35, w_pad=0.18, h_pad=0.18)
    save_figure(fig, out_dir, 'reconstruction_candidates_5targets')
    return summary


def write_csvs(out_dir, selected, method_summary):
    cases_path = out_dir / 'selected_cases.csv'
    with cases_path.open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'sample_idx', 'file_id', 'selection_mode', 'selection_source',
            'fallback_used', 'gt_count', 'min_gt_pair_distance',
            'dista_recall', 'arlr_recall', 'dista_err', 'arlr_err',
            'arlr_precision', 'arlr_det_count',
            'arlr_filtered_det_count', 'arlr_filtered_det_count_gap',
            'arlr_dista_map_diff', 'arlr_dista_filtered_map_diff',
            'recall_gain', 'err_gain', 'from_balanced_pool', 'score',
            'advantage'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in selected:
            writer.writerow({key: case[key] for key in fieldnames})
    info(f'Saved {cases_path}')

    summary_path = out_dir / 'method_summary.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as file:
        fieldnames = [
            'method', 'mean_recall', 'mean_precision',
            'mean_gt_distance', 'mean_det_count'
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in method_summary:
            writer.writerow(row)
    info(f'Saved {summary_path}')


def write_reconstruction_candidate_csv(out_dir, reconstruction_summary):
    candidates_path = out_dir / 'reconstruction_candidate_cases.csv'
    fieldnames = [
        'sample_idx', 'file_id', 'gt_count', 'recon_score',
        'arlr_recall', 'arlr_precision', 'arlr_err',
        'arlr_filtered_det_count', 'arlr_filtered_det_count_gap',
        'arlr_grid_recall', 'arlr_grid_precision', 'arlr_grid_err',
        'arlr_grid_det_count', 'arlr_grid_det_count_gap',
        'dista_recall', 'dista_precision', 'dista_err',
        'dista_filtered_det_count', 'dista_filtered_det_count_gap',
        'dista_grid_recall', 'dista_grid_precision', 'dista_grid_err',
        'dista_grid_det_count', 'dista_grid_det_count_gap',
        'recall_gain', 'err_gain',
        'arlr_dista_filtered_map_diff', 'arlr_dista_map_diff',
        'min_gt_pair_distance'
    ]
    candidates = reconstruction_summary.get('candidate_cases_top30', [])
    with candidates_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({key: row.get(key) for key in fieldnames})
    info(f'Saved {candidates_path}')


def selected_case_records(selected):
    fields = [
        'sample_idx', 'file_id', 'selection_mode', 'selection_source',
        'fallback_used', 'gt_count', 'min_gt_pair_distance',
        'dista_recall', 'arlr_recall', 'dista_err', 'arlr_err',
        'arlr_precision', 'arlr_det_count',
        'arlr_filtered_det_count', 'arlr_filtered_det_count_gap',
        'arlr_dista_map_diff', 'arlr_dista_filtered_map_diff',
        'recall_gain', 'err_gain', 'from_balanced_pool', 'score', 'advantage'
    ]
    return [
        {field: case[field] for field in fields}
        for case in selected
    ]


def main():
    configure_matplotlib_style()
    args = parse_args()
    if args.num_cases < 1:
        raise ValueError('--num-cases must be positive.')
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive.')
    if not 0.0 <= args.background_alpha <= 1.0:
        raise ValueError('--background-alpha must be in [0, 1].')
    if args.clean_localization_style:
        args.background_alpha = 1.0
        args.show_error_lines = False
    out_dir = as_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg, dataset = build_dataset(as_path(args.dataset_config))
    dataset_len = len(dataset)
    indices = parse_indices(args.sample_indices, dataset_len, args.max_scan)
    if not indices:
        raise RuntimeError('No dataset samples were selected for inference.')
    samples = [dataset[idx] for idx in indices]
    dataset_is_mat = bool(
        dataset_cfg.test_dataloader.dataset.get(
            'is_mat',
            dataset_cfg.get('model', {}).get('is_mat', False)))
    samples_by_index = OrderedDict()
    for idx, sample in zip(indices, samples):
        gt_points = read_gt(sample, dataset_is_mat)
        samples_by_index[idx] = {
            'sample_idx': idx,
            'file_id': str(sample.get('file_id', idx)),
            'gt_points': gt_points,
            'lr_image': sample_to_numpy(sample, 'gt_img_11').reshape(11, 11),
            'gt_matrix': sample_to_numpy(sample, 'gt').reshape(33, 33),
        }

    requested_methods = parse_method_names(args.methods)
    if args.skip_ista and 'ISTA' in requested_methods:
        requested_methods.remove('ISTA')
    all_results = OrderedDict()
    metric_cfgs = {}
    method_runtime_info = OrderedDict()
    skipped_reasons = OrderedDict()
    method_configs = OrderedDict()
    for method in requested_methods:
        spec = METHOD_SPECS[method]
        method_configs[method] = {
            'config': spec.get('config', args.dataset_config),
            'checkpoint': spec.get('checkpoint'),
        }

    for method in requested_methods:
        spec = METHOD_SPECS[method]
        try:
            if method == 'ISTA':
                result, metric_cfg, runtime_info = infer_ista(
                    samples, indices, dataset_cfg, args)
            else:
                result, metric_cfg, runtime_info = infer_neural_method(
                    method, spec, samples, indices, args)
        except MethodSkip as exc:
            skipped_reasons[method] = str(exc)
            continue
        except Exception as exc:
            tb = traceback.format_exc()
            warn(
                f'{method}: inference failed with {type(exc).__name__}: '
                f'{exc}\n{tb}')
            skipped_reasons[method] = (
                f'{type(exc).__name__}: {exc}\n{tb}')
            continue
        if result is not None:
            all_results[method] = result
            metric_cfgs[method] = dict(metric_cfg)
            method_runtime_info[method] = runtime_info

    if 'ARLR-Net' not in all_results:
        warn('ARLR-Net results are unavailable. Check its config/checkpoint '
             'before using the generated comparison.')
    if not all_results:
        raise RuntimeError('No methods produced predictions.')

    scored = score_cases(samples_by_index, all_results, args)
    for case in scored:
        case['gt_points'] = samples_by_index[case['sample_idx']]['gt_points']
    selected, selection_info = select_cases(scored, args)
    if not selected:
        return

    draw_arlr_vs_dista(selected, samples_by_index, all_results, out_dir, args)
    draw_arlr_dista_zoom_cases(
        selected, samples_by_index, all_results, out_dir, args)
    method_summary = draw_all_methods_summary(selected, all_results, out_dir, args)
    draw_all_methods_case_grid(selected, samples_by_index, all_results, out_dir, args)
    final_detection_summary = draw_final_detection_comparison_grid(
        samples_by_index,
        all_results,
        out_dir,
        args)
    summary_metric_cfg = (
        metric_cfgs.get('ARLR-Net')
        or next(iter(metric_cfgs.values()), resolve_metric_cfg(dataset_cfg, args)))
    recon_brightness_threshold = summary_metric_cfg.get('brightness_threshold')
    reconstruction_summary = draw_reconstruction_comparison_grid(
        samples_by_index,
        selected,
        all_results,
        out_dir,
        args,
        summary_metric_cfg,
        recon_brightness_threshold)
    reconstruction_false_alarm_summary = draw_reconstruction_false_alarm_grid(
        samples_by_index,
        all_results,
        reconstruction_summary,
        out_dir,
        args)
    reconstruction_aligned_localization_summary = (
        draw_reconstruction_aligned_localization_comparison(
            samples_by_index,
            all_results,
            reconstruction_summary,
            out_dir,
            args))
    paper_representative_localization_summary = (
        draw_paper_representative_localization_grid(
            samples_by_index,
            all_results,
            reconstruction_summary,
            out_dir,
            args))
    reconstruction_preview_summary = draw_reconstruction_candidate_preview_5targets(
        samples_by_index,
        all_results,
        out_dir,
        args,
        reconstruction_summary)
    write_csvs(out_dir, selected, method_summary)
    write_reconstruction_candidate_csv(out_dir, reconstruction_summary)

    arlr_info = method_runtime_info.get('ARLR-Net', {})
    run_summary = {
        'dataset_config': args.dataset_config,
        'num_scanned_samples': len(indices),
        'distance_thr': args.distance_thr,
        'brightness_threshold_override': args.brightness_threshold,
        'plot_matched_only': bool(args.plot_matched_only),
        'show_unmatched_faint': bool(args.show_unmatched_faint),
        'show_error_lines': bool(args.show_error_lines),
        'hide_intensity_background': bool(args.hide_intensity_background),
        'background_cmap': args.background_cmap,
        'background_alpha': args.background_alpha,
        'clean_localization_style': bool(args.clean_localization_style),
        'recon_map_type': args.recon_map_type,
        'recon_methods': list(RECON_METHOD_ORDER),
        'recon_brightness_threshold': recon_brightness_threshold,
        'recon_strict_position': bool(args.recon_strict_position),
        'reconstruction_selection_mode': (
            'arlr_correct_then_dista_distinct'
            if args.selection_mode == 'recon_arlr_distinct'
            else 'arlr_correct_position_strict'),
        'min_recall_gain': args.min_recall_gain,
        'min_error_gain': args.min_error_gain,
        'min_dista_recall': args.min_dista_recall,
        'max_dista_recall': args.max_dista_recall,
        'min_arlr_recall': args.min_arlr_recall,
        'max_arlr_err': args.max_arlr_err,
        'min_arlr_success_recall': args.min_arlr_success_recall,
        'selection_mode': args.selection_mode,
        'random_seed': args.random_seed,
        'random_crowded_only': bool(args.random_crowded_only),
        'selected_sample_indices': [case['sample_idx'] for case in selected],
        'selected_cases': selected_case_records(selected),
        'selection_info': selection_info,
        'final_detection_comparison_grid': final_detection_summary,
        'reconstruction_comparison_grid': reconstruction_summary,
        'reconstruction_false_alarm_grid': reconstruction_false_alarm_summary,
        'reconstruction_aligned_localization_comparison': (
            reconstruction_aligned_localization_summary),
        'paper_representative_localization_grid': (
            paper_representative_localization_summary),
        'reconstruction_candidates_5targets': reconstruction_preview_summary,
        'available_methods': list(all_results.keys()),
        'skipped_methods': [
            method for method in requested_methods if method not in all_results
        ],
        'skipped_reasons': skipped_reasons,
        'method_configs': method_configs,
        'method_runtime_info': method_runtime_info,
        'metric_cfgs': metric_cfgs,
        'arlr_subpixel': {
            'subpixel_refine': bool(arlr_info.get('subpixel_refine', False)),
            'subpixel_offset_in_output': bool(
                arlr_info.get('subpixel_offset_in_output', False)),
            'final_prediction_field': arlr_info.get(
                'final_prediction_field'),
        },
        'notes': [
            'Detections are produced from output["x_final"] exactly as in '
            'CSO_Metrics.process.',
            'For ARLR-Net, subpixel offsets are used only when the selected '
            'config evaluator sets subpixel_refine=True.',
            'reconstruction_comparison_grid shows thresholded grid-level '
            'intensity maps, while final_detection_comparison_grid and '
            'arlr_dista_zoom_cases show final localization results using '
            'x_final plus subpixel_offset for ARLR-Net.',
            'Reconstruction candidate ranking uses grid-level detections '
            'from filtered_matrix without subpixel_offset; this does not '
            'change final localization evaluation or plotting.',
            'reconstruction_false_alarm_grid reuses the '
            'reconstruction_comparison_grid samples and methods, keeps the '
            'same reconstruction-map background, and marks only unmatched '
            'predictions with small red hollow circles.',
            'reconstruction_aligned_localization_comparison uses exactly '
            'the same samples as reconstruction_comparison_grid, but '
            'visualizes final localization results. ARLR-Net detections use '
            'x_final plus subpixel_offset.',
            'paper_representative_localization_grid reuses the '
            'reconstruction_comparison_grid samples and compares GT, '
            'ARLR-Net, DISTA-Net, ISTA, SRFBN, and GGLNet when available. '
            'It shows filtered final intensity maps with ground-truth and '
            'matched-prediction markers, and intentionally omits the bottom '
            'marker legend.',
            'The summary is computed on automatically selected crowded test '
            'cases, not on the full test split unless --max-scan -1 is used.',
        ],
    }
    summary_path = out_dir / 'run_summary.json'
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(run_summary, file, indent=2)
    info(f'Saved {summary_path}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
