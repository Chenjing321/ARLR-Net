import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from mmengine.config import Config
from mmengine.registry import DATASETS

from arlr.datasets import *  # noqa: F401,F403
from arlr.evaluation.metrics.cso_metric import CSO_Metrics
from arlr.evaluation.functional.cso_mean_ap import CSO_tpfp
from arlr.evaluation.functional.cso_mean_ap import average_precision

try:
    from tools.utils import read_targets_from_xml
except ModuleNotFoundError:
    from utils import read_targets_from_xml


DISTANCE_THRS = (0.05, 0.10, 0.15, 0.20, 0.25)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run the traditional ISTA baseline on the CSO test set.')
    parser.add_argument(
        '--config',
        default='configs/c_3/cso_ablation_07_full.py',
        help='Config whose test_dataloader defines the evaluation dataset.')
    parser.add_argument(
        '--phi',
        default='data/sampling_matrix/a_phi_0_3.mat',
        help='Sampling matrix .mat file. It must contain key "phi".')
    parser.add_argument(
        '--out-dir',
        default='work_dir/test/ista_baseline',
        help='Directory for summary files and optional predictions.')
    parser.add_argument('--lambda-value', type=float, default=1.0)
    parser.add_argument('--max-iter', type=int, default=3000)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--brightness-threshold', type=float, default=50.0)
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of test samples. Omit to evaluate all samples.')
    parser.add_argument(
        '--save-vis',
        action='store_true',
        help='Save visualized ISTA output maps for the first samples.')
    parser.add_argument(
        '--max-vis',
        type=int,
        default=80,
        help='Maximum number of visualization images when --save-vis is set.')
    parser.set_defaults(save_pred_txt=True)
    parser.add_argument(
        '--save-pred-txt',
        dest='save_pred_txt',
        action='store_true',
        help='Save per-image prediction txt files. Enabled by default.')
    parser.add_argument(
        '--no-save-pred-txt',
        dest='save_pred_txt',
        action='store_false',
        help='Disable per-image prediction txt output.')
    return parser.parse_args()


def soft_thresholding(x, lambda_):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)


def ista(A, y, lambda_, max_iter=3000, tol=1e-4, A_pinv=None, L=None):
    if A_pinv is None:
        A_pinv = np.linalg.pinv(A)
    if L is None:
        L = np.linalg.norm(A, ord=2) ** 2
    if L <= 0:
        raise ValueError('The Lipschitz constant of A must be positive.')

    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    x = np.dot(A_pinv, y)
    for _ in range(max_iter):
        x_new = soft_thresholding(
            x + A.T.dot(y - A.dot(x)) / L, lambda_ / L)
        if np.linalg.norm(x - x_new) < tol:
            x = x_new
            break
        x = x_new
    return x


def load_phi(phi_path):
    mat = sio.loadmat(phi_path)
    if 'phi' not in mat:
        keys = sorted(key for key in mat.keys() if not key.startswith('__'))
        raise KeyError(
            f'Cannot find key "phi" in {phi_path}. Available keys: {keys}')

    A = np.asarray(mat['phi'], dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f'Expected phi to be 2-D, got shape {A.shape}.')
    return A


def get_first_cso_evaluator_cfg(cfg):
    evaluator = cfg.get('test_evaluator', cfg.get('val_evaluator', {}))
    if isinstance(evaluator, (list, tuple)):
        for metric_cfg in evaluator:
            if metric_cfg.get('type') == 'CSO_Metrics':
                return metric_cfg
        return evaluator[0] if evaluator else {}
    return evaluator or {}


def resolve_c(cfg, A):
    metric_cfg = get_first_cso_evaluator_cfg(cfg)
    c = int(metric_cfg.get('c', cfg.get('model', {}).get('c', 3)))

    output_side = int(round(np.sqrt(A.shape[1])))
    if output_side * output_side != A.shape[1]:
        raise ValueError(
            f'ISTA output dimension must be a square, got {A.shape[1]}.')
    if output_side != 11 * c:
        raise ValueError(
            f'Config c={c} expects output size {(11 * c) ** 2}, '
            f'but phi has {A.shape[1]} columns.')
    return c


def build_test_dataset(cfg):
    if 'test_dataloader' not in cfg:
        raise KeyError('The config does not define test_dataloader.')
    if 'dataset' not in cfg.test_dataloader:
        raise KeyError('cfg.test_dataloader does not define dataset.')
    return DATASETS.build(cfg.test_dataloader.dataset)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def sample_to_y(sample, sample_index):
    for key in ('gt_img_11', 'img_gt', 'gt_img'):
        if key in sample:
            y = tensor_to_numpy(sample[key])
            if y.size != 121:
                raise ValueError(
                    f'Sample {sample_index} field "{key}" has {y.size} '
                    'values, expected the 11x11 observation with 121 values.')
            return y.reshape(-1, 1)
    raise KeyError(
        f'Sample {sample_index} has no 11x11 observation field. '
        'Expected one of: gt_img_11, img_gt, gt_img.')


def get_file_id(sample, sample_index):
    file_id = sample.get('file_id', sample_index)
    if isinstance(file_id, torch.Tensor):
        file_id = file_id.item()
    return str(file_id)


def get_targets(sample, sample_index, is_mat=False):
    if 'ann_path' not in sample:
        raise KeyError(f'Sample {sample_index} does not contain ann_path.')
    return read_targets_from_xml(sample['ann_path'], is_mat=is_mat)


def resolve_is_mat(cfg):
    dataset_cfg = cfg.test_dataloader.dataset
    return bool(dataset_cfg.get('is_mat', cfg.get('model', {}).get(
        'is_mat', False)))


def process_with_cso_metric(metric, x_output, targets):
    output = {
        'x_final': torch.from_numpy(x_output.reshape(1, -1)).float(),
        'targets_GT': [targets],
    }
    metric.process({}, [output])
    return metric.results[-1][1]


def write_prediction_txt(txt_path, detections):
    detections = sorted(detections, key=lambda item: item[2], reverse=True)
    with txt_path.open('w', encoding='utf-8') as file:
        for x_coord, y_coord, brightness in detections:
            file.write(f'{x_coord} {y_coord} {float(brightness)}\n')


def save_visualization(path, matrix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(matrix, cmap='gray')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def compute_cso_metrics(results, distance_thrs):
    gts, preds = zip(*results) if results else ([], [])
    eval_results = OrderedDict()
    mean_aps = []

    for distance_thr in distance_thrs:
        tpfp = [
            CSO_tpfp(det_points, gt_points, distance_thr)
            for det_points, gt_points in zip(preds, gts)
        ]
        num_gts = sum(np.asarray(gt_points).reshape(-1, 3).shape[0]
                      for gt_points in gts)

        non_empty_preds = [pred for pred in preds if len(pred) > 0]
        if not non_empty_preds or num_gts == 0:
            mean_ap = 0.0
        else:
            cls_dets = np.vstack(non_empty_preds)
            sort_inds = np.argsort(-cls_dets[:, -1])
            tp, fp = tuple(zip(*tpfp))
            tp = np.hstack(tp)[sort_inds]
            fp = np.hstack(fp)[sort_inds]
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(np.full(tp.shape, num_gts), eps)
            precisions = tp / np.maximum(tp + fp, eps)
            mean_ap = float(average_precision(recalls, precisions, 'area'))

        mean_aps.append(mean_ap)
        eval_results[f'AP{int(distance_thr * 100):02d}'] = round(mean_ap, 5)

    eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
    eval_results.move_to_end('mAP', last=False)
    return eval_results


def write_summary(out_dir, summary):
    summary_json = out_dir / 'summary.json'
    summary_txt = out_dir / 'summary.txt'

    with summary_json.open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)

    lines = [
        'Traditional Optimization / ISTA baseline',
        f'lambda_value: {summary["lambda_value"]}',
        f'max_iter: {summary["max_iter"]}',
        f'tol: {summary["tol"]}',
        f'brightness_threshold: {summary["brightness_threshold"]}',
        f'num_samples: {summary["num_samples"]}',
        f'average_time_per_sample: {summary["average_time_per_sample"]:.6f}s',
        '',
        f'AP-05: {summary["AP05"]:.5f}',
        f'AP-10: {summary["AP10"]:.5f}',
        f'AP-15: {summary["AP15"]:.5f}',
        f'AP-20: {summary["AP20"]:.5f}',
        f'AP-25: {summary["AP25"]:.5f}',
        f'mAP: {summary["mAP"]:.5f}',
        '',
        'For traditional ISTA, #P/FLOPs/PSNR/SSIM may be reported as "-" '
        'following the original paper table.',
    ]
    summary_txt.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    if args.max_iter < 1:
        raise ValueError('--max-iter must be positive.')
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError('--max-samples must be positive when provided.')
    if args.max_vis < 0:
        raise ValueError('--max-vis must be non-negative.')

    cfg = Config.fromfile(args.config)
    A = load_phi(args.phi)
    c = resolve_c(cfg, A)
    if A.shape[0] != 121:
        raise ValueError(
            f'Expected phi to have 121 rows for 11x11 observations, '
            f'got shape {A.shape}.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_txt_dir = out_dir / 'pred_txt'
    vis_dir = out_dir / 'vis'
    if args.save_pred_txt:
        pred_txt_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_test_dataset(cfg)
    is_mat = resolve_is_mat(cfg)
    num_samples = len(dataset)
    if args.max_samples is not None:
        num_samples = min(num_samples, args.max_samples)

    metric_cfg = get_first_cso_evaluator_cfg(cfg)
    distance_thrs = metric_cfg.get('distance_thrs', DISTANCE_THRS)
    metric = CSO_Metrics(
        distance_thrs=distance_thrs,
        brightness_threshold=args.brightness_threshold,
        c=c,
        subpixel_refine=False)

    A_pinv = np.linalg.pinv(A)
    L = np.linalg.norm(A, ord=2) ** 2
    total_time = 0.0

    for sample_index in range(num_samples):
        sample = dataset[sample_index]
        y = sample_to_y(sample, sample_index)
        targets = get_targets(sample, sample_index, is_mat=is_mat)

        start_time = time.perf_counter()
        x_output = ista(
            A,
            y,
            args.lambda_value,
            max_iter=args.max_iter,
            tol=args.tol,
            A_pinv=A_pinv,
            L=L)
        total_time += time.perf_counter() - start_time

        x_output = np.asarray(x_output, dtype=np.float64).reshape(-1)
        if x_output.size != (11 * c) ** 2:
            raise ValueError(
                f'ISTA returned {x_output.size} values, expected '
                f'{(11 * c) ** 2}.')

        detections = process_with_cso_metric(metric, x_output, targets)
        file_id = get_file_id(sample, sample_index)

        if args.save_pred_txt:
            write_prediction_txt(pred_txt_dir / f'image_{file_id}.txt',
                                 detections)

        if args.save_vis and sample_index < args.max_vis:
            matrix = x_output.reshape(11 * c, 11 * c).copy()
            matrix[matrix < args.brightness_threshold] = 0
            save_visualization(vis_dir / f'image_{file_id}.png', matrix)

    eval_results = compute_cso_metrics(metric.results, distance_thrs)
    average_time = total_time / num_samples if num_samples else 0.0

    summary = {
        'method': 'ISTA',
        'lambda_value': args.lambda_value,
        'max_iter': args.max_iter,
        'tol': args.tol,
        'brightness_threshold': args.brightness_threshold,
        'num_samples': num_samples,
        'AP05': eval_results['AP05'],
        'AP10': eval_results['AP10'],
        'AP15': eval_results['AP15'],
        'AP20': eval_results['AP20'],
        'AP25': eval_results['AP25'],
        'mAP': eval_results['mAP'],
        'average_time_per_sample': average_time,
    }
    write_summary(out_dir, summary)

    print(json.dumps(summary, indent=2))
    print(
        'For traditional ISTA, #P/FLOPs/PSNR/SSIM may be reported as "-" '
        'following the original paper table.')


if __name__ == '__main__':
    main()
