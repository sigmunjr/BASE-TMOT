import os
import time

import numpy as np

import torch
import torchvision
from torch import nn

from sequence_reader import SequenceReader
from utils import write_results, read_vo
from interpolation import dti_sequence_result
from mot_accumulator import MOTSequenceResult, SequenceEvaluator, print_sequence_results
from mot_tracker import create_tracker, get_parameters
from pathlib import Path
import cv2
import torch.nn.functional as F


def xywh2xyxy(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def letterbox_tensor(image: torch.Tensor, new_shape=(640, 640), pad_value=114):
    """
    Resize tensor image with unchanged aspect ratio using padding (letterbox).

    Args:
        image (torch.Tensor): Input tensor image of shape (C, H, W).
        new_shape (tuple): Desired output size as (width, height).
        pad_value (int or float): Constant value for padding.

    Returns:
        torch.Tensor: Resized tensor image of shape (C, new_shape[1], new_shape[0]).
    """
    # Unpack original dimensions and target dimensions
    C, H, W = image.shape
    target_w, target_h = new_shape

    # Compute scale to fit the image within the new shape
    scale = min(target_w / W, target_h / H)
    new_w = int(W * scale)
    new_h = int(H * scale)

    # Resize using bilinear interpolation
    # F.interpolate expects a 4D tensor (batch dimension), so add one.
    image_resized = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
    image_resized = image_resized.squeeze(0)  # back to (C, new_h, new_w)

    # Compute padding amounts to center the image
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    # Pad the resized image
    padded_image = F.pad(image_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
    return padded_image


class YOLOv8Simplified(nn.Module):
    def __init__(self, dtype=torch.float32, device=torch.device('cuda:0'), img_size=(1280, 1024),
                 detections_scale=np.array((1080, 1920))):
        super().__init__()
        self.device = device
        self.traced_model = torch.jit.load('data/last.torchscript').to(device)
        self.dtype = dtype
        self.iou_thresh = 0.7
        self.img_size = torch.tensor(img_size).repeat(2).to(device)
        self.detections_scale = torch.tensor(detections_scale[::-1].copy()).to(device).repeat(2)

    def forward(self, img, is_bgr=False):
        if is_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_scale = torch.tensor(img.shape[1]).to(self.device)
        img = torch.tensor(img, device=self.device).permute(2, 0, 1).to(self.dtype)
        img = letterbox_tensor(img, new_shape=(1280, 1024))[None]
        img = img.to(self.device).to(self.dtype) / 255
        x = self.traced_model(img)[0]
        xc = x[4] >= 0.1
        x = x.transpose(-1, -2)
        x = x[xc]
        x[..., :4] = xywh2xyxy(x[..., :4])
        box, scores = x[:, :4], x[:, 4]
        indices = torchvision.ops.nms(box, scores, self.iou_thresh)
        out = x[indices]
        out[:, :4] = out[:, :4] * self.detections_scale / self.img_size
        out[:, :4] = xyxy2xywh(out[:, :4] + 1)
        return out.cpu()


def get_img_folders(folder_path, img_folder_name='thermal'):
    img_folders = []
    for root, dirs, files in os.walk(folder_path):
        for d in dirs:
            if d == img_folder_name:
                base_root = os.path.dirname(root)
                name = os.path.basename(root)
                img_folders.append((base_root, name, d, sorted(os.listdir(os.path.join(root, d)))))
    return img_folders


def timestamp_from_filename(filename):
    return int(filename.split('.')[0])


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def run_live_image_folder(folder_path, img_folder_name='thermal', visualize=False,
                          results_path='./data/results/tmot_base', write_to_file=True):

    img_folders = get_img_folders(folder_path, img_folder_name)
    parameters = get_parameters('tmot_')
    parameters['R'] = torch.tensor([[10.9158, 0.1866, -0.1089, -0.5445],
                                    [0.1866, 10.1854, -0.5839, -0.9501],
                                    [-0.1089, -0.5839, 20.3491, 3.2942],
                                    [-0.5445, -0.9501, 3.2942, 50.5327]], dtype=torch.float64)
    detection_size = np.array((1080, 1920))
    det_model = YOLOv8Simplified(detections_scale=detection_size)
    tracker = create_tracker(parameters, detection_size, preprocess=False)
    names = []
    for root, name, img_folder, files in img_folders:
        print(f"Processing folder {name}")
        prev_timestamp = timestamp_from_filename(files[0]) - 1 / 30
        begin = time.time()
        tracker.reset()
        names.append(name)
        sequence_result = MOTSequenceResult(name)
        img_buffer = []
        for frame_id, f in enumerate(files, start=1):
            img = cv2.imread(os.path.join(root, name, img_folder, f))
            img_buffer.append(img)
            detections = det_model(img, is_bgr=True)
            timestamp = timestamp_from_filename(f)
            update_s = (timestamp - prev_timestamp) / 1e9
            prev_timestamp = timestamp

            h, w = img.shape[:2]

            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = tracker.track(
                frame_id=frame_id,
                detections=detections,
                detection_size=detection_size,
                in_size_hw=(h, w),
                update_s=update_s
            )
            if not frame_id:
                continue

            if visualize:
                show_img = img_buffer.pop(0)
                show_img_cv(show_img, track_ids, track_tlwhs)

            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        while data := tracker.track_from_buffer((h, w)):
            if visualize:
                show_img = img_buffer.pop(0)
                show_img_cv(show_img, track_ids, track_tlwhs)
            if not data[0]:
                continue
            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = data
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        end = time.time()
        print(f'took {(end - begin) * 1e3:.2f}ms')
        if write_to_file:
            result_filename = os.path.join(results_path, '{}.txt'.format(name))
            write_results(result_filename, sequence_result)


def show_img_cv(show_img, track_ids, track_tlwhs):
    if track_ids is not None:
        for tlwh, id in zip(track_tlwhs, track_ids):
            color = get_color(id.item())
            p1, p2 = tlwh[:2].astype(int).tolist(), (tlwh[:2] + tlwh[2:]).astype(int).tolist()
            cv2.rectangle(show_img, p1, p2, color)
            cv2.putText(show_img, str(id), p1, cv2.FONT_HERSHEY_PLAIN,
                        1, color, thickness=1)
    cv2.imshow("BASE", show_img)
    cv2.waitKey(1)


def run_with_live_detection_in_mot_folder(base_folder):
    write_to_file = True
    get_metrics = True
    results_path = './data/results/tmot_base'

    if write_to_file:
        Path(results_path).mkdir(parents=True, exist_ok=True)

    reader = SequenceReader(
        f'{base_folder}/train/seq*',
        detections_filename='det_yolov8s_1280_mtmmc',
        # im_dir='thermal',
    )
    sequence_results = []
    detection_size = np.array((1024, 1280))  # tracker.detection_size
    det_model = YOLOv8Simplified(detections_scale=detection_size)
    parameters = get_parameters('tmot_')
    parameters['R'] = torch.tensor([[10.9158, 0.1866, -0.1089, -0.5445],
                                    [0.1866, 10.1854, -0.5839, -0.9501],
                                    [-0.1089, -0.5839, 20.3491, 3.2942],
                                    [-0.5445, -0.9501, 3.2942, 50.5327]], dtype=torch.float64)
    print(parameters)
    tracker = create_tracker(parameters, detection_size, preprocess=False)
    names = []
    full_runtimes = []
    track_times = []
    det_times = []
    for (seq, info) in reader:
        valid_indices = seq.valid_indices
        img_files = seq.img_files
        print(f"Processing sequence {info['name']}")
        name = info['name']
        # seq_vo = vo[name]
        names.append(name)
        sequence_result = MOTSequenceResult(name)
        sequence_evaluator = SequenceEvaluator(name, metric_names=('CLEAR', 'Identity'))

        if seq.groundtruth_frame is None:
            get_metrics = False
            print('Found no ground truth, running without metrics')

        if get_metrics:
            sequence_evaluator.add_gt(MOTSequenceResult(name).from_pd_dataframe(seq.gt_path, valid_indices))

        tracker.reset()

        w, h = info['imWidth'], info['imHeight']
        update_s = info['update_s']

        begin = time.time()

        for seq_frame_id, (detections_unfiltered, gt) in zip(valid_indices,
                                                             seq) if valid_indices is not None else enumerate(
            seq,
            start=1):
            img = cv2.imread(img_files[seq_frame_id - 1])
            det_start = time.time()
            detections = det_model(img, is_bgr=True)
            det_times.append(time.time() - det_start)

            track_start = time.time()
            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = tracker.track(
                frame_id=seq_frame_id,
                detections=detections,
                detection_size=detection_size,
                in_size_hw=(h, w),
                update_s=update_s
            )
            track_times.append(time.time() - track_start)

            if not frame_id:
                continue
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        while data := tracker.track_from_buffer((h, w)):
            if not data[0]:
                continue
            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = data
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        end = time.time()
        full_runtimes.append(end - begin)
        print(f'took {(end - begin) * 1e3:.2f}ms')

        if parameters['interpolate']:
            sequence_result = dti_sequence_result(sequence_result, None, (h, w))

        if get_metrics:
            sequence_evaluator.add_results(sequence_result)
            sequence_results.append(sequence_evaluator)
            print(
                f'MOTA: {sequence_evaluator.sequence_results["MOTA"]}'
                f' IDF1: {sequence_evaluator.sequence_results["IDF1"]}'
            )
        if write_to_file:
            result_filename = os.path.join(results_path, '{}.txt'.format(name))
            write_results(result_filename, sequence_result)
    if get_metrics:
        print_sequence_results(sequence_results)
    print(f'mean full runtime: {np.mean(full_runtimes)} mean det time: {np.mean(det_times)} mean track time: {np.mean(track_times)}')
    print('done')


def run_stuff(base_folder):
    get_metrics = True
    write_to_file = True

    results_path = './data/results/tmot_base'

    if write_to_file:
        Path(results_path).mkdir(parents=True, exist_ok=True)

    reader = SequenceReader(
        f'{base_folder}/valval/seq*',
        detections_filename='det_yolov8s_1280_mtmmc',
    )
    sequence_results = []
    # detection_size = np.array((1080, 1920))  # tracker.detection_size
    detection_size = np.array((1024, 1280))  # tracker.detection_size
    parameters = get_parameters('tmot_')
    parameters['R'] = torch.tensor([[10.9158, 0.1866, -0.1089, -0.5445],
                                    [0.1866, 10.1854, -0.5839, -0.9501],
                                    [-0.1089, -0.5839, 20.3491, 3.2942],
                                    [-0.5445, -0.9501, 3.2942, 50.5327]], dtype=torch.float64)
    print(parameters)
    tracker = create_tracker(parameters, detection_size)

    names = []
    for (seq, info) in reader:
        valid_indices = seq.valid_indices
        print(f"Processing sequence {info['name']}")
        name = info['name']
        names.append(name)
        sequence_result = MOTSequenceResult(name)
        sequence_evaluator = SequenceEvaluator(name, metric_names=('CLEAR', 'Identity'))

        if seq.groundtruth_frame is None:
            get_metrics = False
            print('Found no ground truth, running without metrics')

        if get_metrics:
            sequence_evaluator.add_gt(MOTSequenceResult(name).from_pd_dataframe(seq.gt_path, valid_indices))

        tracker.reset()

        w, h = info['imWidth'], info['imHeight']
        update_s = info['update_s']

        begin = time.time()

        for seq_frame_id, (detections_unfiltered, gt) in zip(valid_indices,
                                                             seq) if valid_indices is not None else enumerate(
            seq,
            start=1):
            detections = []
            for detection in detections_unfiltered:
                if detection.score >= parameters['filter_detection_threshold']:
                    detections.append(detection)

            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = tracker.track(
                frame_id=seq_frame_id,
                detections=detections,
                detection_size=detection_size,
                in_size_hw=(h, w),
                update_s=update_s
            )

            if not frame_id:
                continue
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        while data := tracker.track_from_buffer((h, w)):
            if not data[0]:
                continue
            frame_id, track_confidences, track_llrs, track_ids, track_labels, track_tlwhs = data
            sequence_result.add_frame_data(
                frame_id=frame_id,
                np_confidences=track_confidences,
                np_track_ids=track_ids,
                np_labels=track_labels,
                np_tlwhs=track_tlwhs
            )

        end = time.time()
        print(f'took {(end - begin) * 1e3:.2f}ms')

        if parameters['interpolate']:
            sequence_result = dti_sequence_result(sequence_result, None, (h, w))

        if get_metrics:
            sequence_evaluator.add_results(sequence_result)
            sequence_results.append(sequence_evaluator)
            print(
                f'MOTA: {sequence_evaluator.sequence_results["MOTA"]}'
                f' IDF1: {sequence_evaluator.sequence_results["IDF1"]}'
            )
        if write_to_file:
            result_filename = os.path.join(results_path, '{}.txt'.format(name))
            write_results(result_filename, sequence_result)
    if get_metrics:
        print_sequence_results(sequence_results)
    print('done')


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        print('Usage:\n--------------')
        print(f'python base_tmot.py <data_folder train/val/test>')
        sys.exit(1)
    with torch.no_grad():
        # run_stuff(base_folder)
        # run_with_live_detection_in_mot_folder(base_folder)
        run_live_image_folder(base_folder, visualize=True)
