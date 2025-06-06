#!/usr/bin/env python3

import logging

import ipdb
import numpy as np
import os
import random
import time
from collections import defaultdict
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from slowfast.utils.env import pathmgr

from . import transform as transform

logger = logging.getLogger(__name__)


def retry_load_images(image_paths, retry=10, backend="pytorch"):
    """
    This function is to load images with support of retrying for failed load.

    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.

    Returns:
        imgs (list): list of loaded images.
    """
    for i in range(retry):
        imgs = []
        for image_path in image_paths:
            with pathmgr.open(image_path, "rb") as f:
                img_str = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(img_str, flags=cv2.IMREAD_COLOR)
            imgs.append(img)

        if all(img is not None for img in imgs):
            if backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.

    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames

    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq


def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.DATA.REVERSE_INPUT_CHANNEL:
        frames = frames[[2, 1, 0], :, :, :]
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
    gaze_loc=None,
    use_padding=False,
    pad_value=0,  # 0 for black padding, or use 'mean' for mean padding
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        random_horizontal_flip (bool): whether to apply random horizontal flip.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
        gaze_loc (tensor, optional): Gaze location coordinates to transform along with frames.
        use_padding (bool): If True, use padding instead of cropping to achieve target size.
        pad_value (float or str): Padding value. Use 0 for black, 'mean' for mean padding.
        
    Returns:
        frames (tensor): spatially sampled frames.
        gaze_loc (tensor, optional): transformed gaze locations if provided.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    
    if use_padding:
        # Use padding approach instead of cropping
        frames, gaze_loc = _spatial_sampling_with_padding(
            frames, 
            crop_size, 
            spatial_idx, 
            gaze_loc, 
            pad_value,
            random_horizontal_flip
        )
    else:
        # Original cropping approach
        if spatial_idx == -1:
            if aspect_ratio is None and scale is None:
                frames, _ = transform.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )
                if gaze_loc is None:
                    frames, _ = transform.random_crop(frames, crop_size)
                else:
                    frames, gaze_loc = transform.random_crop_gaze(frames, crop_size, gaze_loc=gaze_loc)
            else:  # have not been modified to gaze aug
                transform_func = (
                    transform.random_resized_crop_with_shift
                    if motion_shift
                    else transform.random_resized_crop
                )
                frames = transform_func(
                    images=frames,
                    target_height=crop_size,
                    target_width=crop_size,
                    scale=scale,
                    ratio=aspect_ratio,
                )
            if random_horizontal_flip:
                if gaze_loc is None:
                    frames, _ = transform.horizontal_flip(0.5, frames)
                else:
                    frames, gaze_loc = transform.horizontal_flip_gaze(0.5, frames, gaze_loc=gaze_loc)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
            frames, _ = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
            if gaze_loc is None:
                frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
            else:
                frames, gaze_loc = transform.uniform_crop_gaze(frames, crop_size, spatial_idx, gaze_loc=gaze_loc)

    if gaze_loc is None:
        return frames
    else:
        return frames, gaze_loc


def _spatial_sampling_with_padding(frames, target_size, spatial_idx, gaze_loc=None, pad_value=0, random_horizontal_flip=True):
    """
    Resize frames to target size using padding instead of cropping.
    
    Args:
        frames (tensor): Input frames with shape [T, H, W, C] or [C, T, H, W]
        target_size (int): Target square size
        spatial_idx (int): Spatial sampling index (-1 for random, 0-2 for deterministic)
        gaze_loc (tensor, optional): Gaze coordinates to transform
        pad_value (float or str): Padding value
        random_horizontal_flip (bool): Whether to apply horizontal flip
        
    Returns:
        frames (tensor): Padded and resized frames
        gaze_loc (tensor, optional): Transformed gaze coordinates
    """
    # Handle different input formats
    if frames.dim() == 4:
        if frames.shape[0] == 3:  # [C, T, H, W]
            frames = frames.permute(1, 2, 3, 0)  # -> [T, H, W, C]
            input_format = 'CTHW'
        else:  # [T, H, W, C]
            input_format = 'THWC'
    else:
        raise ValueError(f"Unexpected frame dimensions: {frames.shape}")
    
    T, H, W, C = frames.shape
    
    # Calculate scaling to fit the larger dimension
    scale = target_size / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    
    # Resize frames
    # Convert to [T*C, H, W] for interpolation
    frames_reshaped = frames.permute(0, 3, 1, 2).reshape(T * C, H, W).unsqueeze(1)  # [T*C, 1, H, W]
    frames_resized = F.interpolate(frames_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False)
    frames_resized = frames_resized.squeeze(1).reshape(T, C, new_H, new_W).permute(0, 2, 3, 1)  # -> [T, H, W, C]
    
    # Calculate padding
    pad_H = target_size - new_H
    pad_W = target_size - new_W
    
    if spatial_idx == -1:
        # Random padding placement
        if pad_H > 0:
            pad_top = torch.randint(0, pad_H + 1, (1,)).item()
            pad_bottom = pad_H - pad_top
        else:
            pad_top = pad_bottom = 0
            
        if pad_W > 0:
            pad_left = torch.randint(0, pad_W + 1, (1,)).item()
            pad_right = pad_W - pad_left
        else:
            pad_left = pad_right = 0
    else:
        # Deterministic padding based on spatial_idx
        if spatial_idx == 0:  # Top/Left
            pad_top, pad_bottom = 0, pad_H
            pad_left, pad_right = 0, pad_W
        elif spatial_idx == 1:  # Center
            pad_top, pad_bottom = pad_H // 2, pad_H - pad_H // 2
            pad_left, pad_right = pad_W // 2, pad_W - pad_W // 2
        elif spatial_idx == 2:  # Bottom/Right
            pad_top, pad_bottom = pad_H, 0
            pad_left, pad_right = pad_W, 0
    
    # Determine padding value
    if pad_value == 'mean':
        pad_val = frames_resized.mean().item()
    else:
        pad_val = pad_value
    
    # Apply padding - need to permute for F.pad which expects [..., H, W]
    frames_padded = frames_resized.permute(0, 3, 1, 2)  # [T, C, H, W]
    frames_padded = F.pad(
        frames_padded, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', 
        value=pad_val
    )
    frames_padded = frames_padded.permute(0, 2, 3, 1)  # Back to [T, H, W, C]
    
    # Transform gaze coordinates if provided
    if gaze_loc is not None:
        # Scale gaze coordinates
        gaze_loc_scaled = gaze_loc.clone()
        gaze_loc_scaled[:, 0] = gaze_loc[:, 0] * scale + pad_left  # x coordinates
        gaze_loc_scaled[:, 1] = gaze_loc[:, 1] * scale + pad_top   # y coordinates
        
        # Clip to valid range
        gaze_loc_scaled[:, 0] = torch.clamp(gaze_loc_scaled[:, 0], 0, target_size - 1)
        gaze_loc_scaled[:, 1] = torch.clamp(gaze_loc_scaled[:, 1], 0, target_size - 1)
    else:
        gaze_loc_scaled = None
    
    # Apply horizontal flip if requested
    if random_horizontal_flip and (spatial_idx == -1):
        if torch.rand(1) < 0.5:
            frames_padded = torch.flip(frames_padded, dims=[2])  # Flip width dimension
            if gaze_loc_scaled is not None:
                gaze_loc_scaled[:, 0] = target_size - 1 - gaze_loc_scaled[:, 0]  # Flip x coordinates
    
    # Convert back to original format if needed
    if input_format == 'CTHW':
        frames_padded = frames_padded.permute(3, 0, 1, 2)  # [T, H, W, C] -> [C, T, H, W]
    
    return frames_padded, gaze_loc_scaled


# Example usage in your data loading pipeline:
def apply_spatial_sampling_with_padding(frames, labels, cfg, spatial_sample_index):
    """
    Apply spatial sampling with padding option based on config.
    
    Args:
        frames: Input video frames
        labels: Gaze location labels
        cfg: Configuration object
        spatial_sample_index: Spatial sampling index
        
    Returns:
        Processed frames and labels
    """
    # Get padding settings from config (you would add these to your config)
    use_padding = getattr(cfg.DATA, 'USE_PADDING', False)
    pad_value = getattr(cfg.DATA, 'PAD_VALUE', 0)  # 0 for black, 'mean' for mean padding
    
    # Calculate scales
    min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
    max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
    crop_size = cfg.DATA.TRAIN_CROP_SIZE
    
    # Normalize frames
    frames = tensor_normalize(frames, cfg.DATA.MEAN, cfg.DATA.STD)
    
    # T H W C -> C T H W (if needed by your pipeline)
    frames = frames.permute(3, 0, 1, 2)
    
    # Perform spatial sampling with padding option
    frames, labels = spatial_sampling(
        frames,
        gaze_loc=labels,
        spatial_idx=spatial_sample_index,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
        use_padding=use_padding,
        pad_value=pad_value,
    )
    
    return frames, labels


def as_binary_vector(labels, num_classes):
    """
    Construct binary label vector given a list of label indices.
    Args:
        labels (list): The input label list.
        num_classes (int): Number of classes of the label vector.
    Returns:
        labels (numpy array): the resulting binary vector.
    """
    label_arr = np.zeros((num_classes,))

    for lbl in set(labels):
        label_arr[lbl] = 1.0
    return label_arr


def aggregate_labels(label_list):
    """
    Join a list of label list.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): The joint list of all lists in input.
    """
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def convert_to_video_level_labels(labels):
    """
    Aggregate annotations from all frames of a video to form video-level labels.
    Args:
        labels (list): The input label list.
    Returns:
        labels (list): Same as input, but with each label replaced by
        a video-level one.
    """
    for video_id in range(len(labels)):
        video_level_labels = aggregate_labels(labels[video_id])
        for i in range(len(labels[video_id])):
            labels[video_id][i] = video_level_labels
    return labels


def load_image_lists(frame_list_file, prefix="", return_list=False):
    """
    Load image paths and labels from a "frame list".
    Each line of the frame list contains:
    `original_vido_id video_id frame_id path labels`
    Args:
        frame_list_file (string): path to the frame list.
        prefix (str): the prefix for the path.
        return_list (bool): if True, return a list. If False, return a dict.
    Returns:
        image_paths (list or dict): list of list containing path to each frame.
            If return_list is False, then return in a dict form.
        labels (list or dict): list of list containing label of each frame.
            If return_list is False, then return in a dict form.
    """
    image_paths = defaultdict(list)
    labels = defaultdict(list)
    with pathmgr.open(frame_list_file, "r") as f:
        assert f.readline().startswith("original_vido_id")
        for line in f:
            row = line.split()
            # original_vido_id video_id frame_id path labels
            assert len(row) == 5
            video_name = row[0]
            if prefix == "":
                path = row[3]
            else:
                path = os.path.join(prefix, row[3])
            image_paths[video_name].append(path)
            frame_labels = row[-1].replace('"', "")
            if frame_labels != "":
                labels[video_name].append(
                    [int(x) for x in frame_labels.split(",")]
                )
            else:
                labels[video_name].append([])

    if return_list:
        keys = image_paths.keys()
        image_paths = [image_paths[key] for key in keys]
        labels = [labels[key] for key in keys]
        return image_paths, labels
    return dict(image_paths), dict(labels)


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def get_random_sampling_rate(long_cycle_sampling_rate, sampling_rate):
    """
    When multigrid training uses a fewer number of frames, we randomly
    increase the sampling rate so that some clips cover the original span.
    """
    if long_cycle_sampling_rate > 0:
        assert long_cycle_sampling_rate >= sampling_rate
        return random.randint(sampling_rate, long_cycle_sampling_rate)
    else:
        return sampling_rate


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def create_sampler(dataset, shuffle, cfg):
    """
    Create sampler for the given dataset.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
        shuffle (bool): set to ``True`` to have the data reshuffled
            at every epoch.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        sampler (Sampler): the created sampler.
    """
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    return sampler


def loader_worker_init_fn(dataset):
    """
    Create init function passed to pytorch data loader.
    Args:
        dataset (torch.utils.data.Dataset): the given dataset.
    """
    return None
