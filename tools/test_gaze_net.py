#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""
import numpy as np
import os
import pickle
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.utils.metrics as metrics
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter, TestGazeMeter
from slowfast.utils.utils import frame_softmax
from slowfast.datasets.gaze_exporter import export_gaze_annotations

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_test(test_loader, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestGazeMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels_hm, video_idx, meta) in enumerate(test_loader):
        print(f"labels_hm: {labels_hm}")
        print(f"video_idx: {video_idx}")
        print(f"meta: {meta}")
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Transfer the data to the current GPU device.
            # labels = labels.cuda()
            labels_hm = labels_hm.cuda()
            video_idx = video_idx.cuda()

        test_meter.data_toc()

        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
            ori_boxes = (
                ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
            )
            metadata = (
                metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
            )

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(preds, ori_boxes, metadata)
            test_meter.log_iter_stats(None, cur_iter)
        else:
            # Perform the forward pass.
            preds = model(inputs)
            # preds, glc = model(inputs, return_glc=True)  # used to visualization glc correlation

            preds = frame_softmax(preds, temperature=2)  # KLDiv
            analyze_and_visualize_temporal_gaze(inputs, preds)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds, labels_hm, video_idx = du.all_gather([preds, labels_hm, video_idx])

            # PyTorch
            if cfg.NUM_GPUS:  # compute on cpu
                preds = preds.cpu()
                # labels = labels.cpu()
                labels_hm = labels_hm.cpu()
                video_idx = video_idx.cpu()

            preds_rescale = preds.detach().view(preds.size()[:-2] + (preds.size(-1) * preds.size(-2),))
            preds_rescale = (preds_rescale - preds_rescale.min(dim=-1, keepdim=True)[0]) / (preds_rescale.max(dim=-1, keepdim=True)[0] - preds_rescale.min(dim=-1, keepdim=True)[0] + 1e-6)
            preds_rescale = preds_rescale.view(preds.size())
            # f1, recall, precision, threshold = metrics.adaptive_f1(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)
            # auc = metrics.auc(preds_rescale, labels_hm, labels, dataset=cfg.TEST.DATASET)

            test_meter.iter_toc()

            # # Update and log stats.
            test_meter.update_stats(preds=preds_rescale, labels_hm=labels_hm)  # If running  on CPU (cfg.NUM_GPUS == 0), use 1 to represent 1 CPU.
            test_meter.log_iter_stats(cur_iter)

        test_meter.iter_tic()

    # Log epoch stats and print the final testing results.
    if True:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
        if writer is not None:
            writer.plot_eval(preds=all_preds, labels=all_labels)

        if True:
            save_path = os.path.join("/home/ville/GLC/output/out.pkl")

            if du.is_root_proc():
                with pathmgr.open(save_path, "wb") as f:
                    pickle.dump([all_preds, all_labels], f)

            logger.info("Successfully saved prediction results to {}".format(save_path))

    test_meter.finalize_metrics()
    return test_meter


def test(cfg):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (test_loader.dataset.num_videos % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS) == 0)
        # Create meters for multi-view testing.
        test_meter = TestGazeMeter(
            num_videos=test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            num_clips=cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            num_cls=cfg.MODEL.NUM_CLASSES,
            overall_iters=len(test_loader),
            dataset=cfg.TEST.DATASET
        )

    # Set up writer for logging to Tensorboard format.
    # if cfg.TENSORBOARD.ENABLE and du.is_master_proc(cfg.NUM_GPUS * cfg.NUM_SHARDS):
    writer = tb.TensorboardWriter(cfg)
    # else:
    #     writer = None

    # Perform multi-view test on the entire dataset.
    test_meter = perform_test(test_loader, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()

    logger.info("Testing finished!")









###### VISUALISE

import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom


def find_gaze_center_point(heatmap, method='weighted_centroid'):
    """
    Find the center point of gaze prediction from a heatmap.
    
    Args:
        heatmap (np.ndarray): 2D heatmap of gaze predictions
        method (str): Method to use for finding center point
            - 'weighted_centroid': Weighted average of all pixels
            - 'gaussian_fit': Fit Gaussian and find center
            - 'max_point': Simple maximum (not recommended)
            - 'top_k_centroid': Centroid of top k% of pixels
    
    Returns:
        center_y, center_x: Coordinates of the gaze center point
        confidence: Confidence measure for the prediction
    """
    if method == 'weighted_centroid':
        # Use all pixels weighted by their probability
        y_coords, x_coords = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        
        # Normalize weights
        weights = heatmap / (heatmap.sum() + 1e-8)
        
        center_y = np.sum(y_coords * weights)
        center_x = np.sum(x_coords * weights)
        confidence = heatmap.max()
        
    elif method == 'top_k_centroid':
        # Use only the top 10% of pixels for centroid calculation
        threshold = np.percentile(heatmap, 90)
        mask = heatmap >= threshold
        
        if mask.sum() > 0:
            y_coords, x_coords = np.where(mask)
            weights = heatmap[mask]
            weights = weights / weights.sum()
            
            center_y = np.sum(y_coords * weights)
            center_x = np.sum(x_coords * weights)
            confidence = weights.max()
        else:
            # Fallback to max point
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            center_y, center_x = max_idx
            confidence = heatmap.max()
    
    elif method == 'gaussian_fit':
        from scipy.optimize import curve_fit
        
        def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = coords
            return amplitude * np.exp(-(((x-x0)**2)/(2*sigma_x**2) + ((y-y0)**2)/(2*sigma_y**2)))
        
        try:
            y_coords, x_coords = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
            coords = np.vstack([x_coords.ravel(), y_coords.ravel()])
            
            # Initial guess
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            initial_guess = [heatmap.max(), max_idx[1], max_idx[0], 5, 5]
            
            popt, _ = curve_fit(gaussian_2d, coords, heatmap.ravel(), p0=initial_guess)
            center_x, center_y = popt[1], popt[2]
            confidence = popt[0]
        except:
            # Fallback to weighted centroid
            return find_gaze_center_point(heatmap, 'weighted_centroid')
    
    elif method == 'max_point':
        max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        center_y, center_x = max_idx
        confidence = heatmap.max()
    
    return center_y, center_x, confidence

def analyze_channels_as_temporal(preds):
    """
    Analyze if the 8 channels represent temporal predictions for 8 frames.
    
    Args:
        preds: Tensor with shape [batch, 1, 8, H, W] where 8 might be temporal
    """
    print("=== ANALYZING CHANNELS AS TEMPORAL FRAMES ===")
    
    batch_0 = preds[0, 0, :, :, :].cpu().numpy()  # [8, H, W]
    
    print(f"Analyzing 8 channels as potential temporal frames:")
    centers = []
    max_values = []
    
    for channel_idx in range(8):
        heatmap = batch_0[channel_idx]  # [H, W]
        
        # Find center and max value
        max_val = heatmap.max()
        center_y, center_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        
        # Also calculate weighted centroid
        y_coords, x_coords = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        weights = heatmap / (heatmap.sum() + 1e-8)
        centroid_y = np.sum(y_coords * weights)
        centroid_x = np.sum(x_coords * weights)
        
        centers.append((center_x, center_y, centroid_x, centroid_y))
        max_values.append(max_val)
        
        print(f"   Channel {channel_idx}: max={max_val:.2e} at ({center_x}, {center_y}), "
              f"centroid=({centroid_x:.1f}, {centroid_y:.1f})")
    
    # Check for temporal variation
    center_positions = np.array([(c[0], c[1]) for c in centers])
    centroid_positions = np.array([(c[2], c[3]) for c in centers])
    
    # Calculate variance in positions
    center_var = np.var(center_positions, axis=0)
    centroid_var = np.var(centroid_positions, axis=0)
    
    print(f"\nPosition variance analysis:")
    print(f"   Max position variance: x={center_var[0]:.2f}, y={center_var[1]:.2f}")
    print(f"   Centroid variance: x={centroid_var[0]:.2f}, y={centroid_var[1]:.2f}")
    
    if center_var.sum() > 5 or centroid_var.sum() > 5:
        print("   → HIGH VARIANCE: Channels likely represent different temporal frames! ✅")
        temporal_interpretation = True
    else:
        print("   → LOW VARIANCE: Channels might be spatial features, not temporal")
        temporal_interpretation = True  # Let's assume temporal for now
    
    return temporal_interpretation, centers, max_values

def visualize_temporal_gaze_predictions(inputs, preds, save_path=None, alpha=0.5, max_frames_to_show=4):
    """
    Visualize gaze predictions treating the 8 channels as temporal predictions.
    
    Args:
        inputs: Input tensor [batch, 3, 8, 256, 256]
        preds: Predictions [batch, 1, 8, 64, 64] where 8 channels = 8 temporal frames
        save_path: Optional save path
        alpha: Overlay transparency
        max_frames_to_show: Maximum number of frames to visualize
    """
    # Check if channels should be interpreted as temporal
    temporal_interpretation, centers, max_values = analyze_channels_as_temporal(preds)    
    # Treat channels as temporal: reshape [B, 1, 8, H, W] → [B, 8, 1, H, W]
    preds_temporal = preds.permute(0, 2, 1, 3, 4)  # [batch, 8, 1, 64, 64]
    
    print(f"\n✅ TREATING CHANNELS AS TEMPORAL")
    print(f"Reshaped predictions from {preds.shape} to {preds_temporal.shape}")
    print(f"Now: [batch, temporal=8, channels=1, H, W]")
    
    # Extract input tensor
    input_tensor = inputs[0] if isinstance(inputs, list) else inputs
    
    # Convert to numpy
    if isinstance(input_tensor, torch.Tensor):
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()
        input_tensor = input_tensor.numpy()
    
    if isinstance(preds_temporal, torch.Tensor):
        if preds_temporal.is_cuda:
            preds_temporal = preds_temporal.cpu()
        preds_temporal = preds_temporal.numpy()
    
    batch_size, img_channels, temporal_frames, img_H, img_W = input_tensor.shape
    pred_batch, pred_temporal, pred_channels, pred_H, pred_W = preds_temporal.shape
    
    print(f"Input: [batch={batch_size}, channels={img_channels}, temporal={temporal_frames}, H={img_H}, W={img_W}]")
    print(f"Predictions: [batch={pred_batch}, temporal={pred_temporal}, channels={pred_channels}, H={pred_H}, W={pred_W}]")
    
    # Select frames to show
    frame_indices = list(range(min(temporal_frames, pred_temporal, max_frames_to_show)))
    
    # Create colormap
    colors = ['black', 'darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'white']
    cmap = LinearSegmentedColormap.from_list('gaze', colors, N=256)
    
    # Setup subplots
    n_frames = len(frame_indices)
    n_cols = min(n_frames, 4)
    n_rows = batch_size * ((n_frames + n_cols - 1) // n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    all_gaze_centers = []
    
    for batch_idx in range(batch_size):
        batch_centers = []
        
        for frame_idx in frame_indices:
            if plot_idx >= len(axes):
                break
            
            # Get input image
            img = input_tensor[batch_idx, :, frame_idx, :, :]  # [3, 256, 256]
            img = np.transpose(img, (1, 2, 0))  # [256, 256, 3]
            
            # Normalize image
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_normalized = (img - img_min) / (img_max - img_min)
            else:
                img_normalized = img
            img_normalized = np.clip(img_normalized, 0, 1)
            
            # Get corresponding temporal prediction
            if batch_idx < pred_batch and frame_idx < pred_temporal:
                pred_heatmap = preds_temporal[batch_idx, frame_idx, 0, :, :]  # [H, W]
                print(f"Frame {frame_idx}: prediction max = {pred_heatmap.max():.2e}")
            else:
                pred_heatmap = np.zeros((pred_H, pred_W))
                print(f"Frame {frame_idx}: no prediction available")
            
            # Find gaze center
            center_y_orig, center_x_orig, confidence = find_gaze_center_point(pred_heatmap, 'gaussian_fit')
            
            # Scale to image coordinates
            scale_y = img_H / pred_H
            scale_x = img_W / pred_W
            center_y_scaled = center_y_orig * scale_y
            center_x_scaled = center_x_orig * scale_x
            
            batch_centers.append({
                'frame': frame_idx,
                'center_original': (center_x_orig, center_y_orig),
                'center_scaled': (center_x_scaled, center_y_scaled),
                'confidence': confidence
            })
            
            # Resize heatmap
            if pred_H != img_H or pred_W != img_W:
                from scipy.ndimage import zoom
                zoom_factor = (img_H / pred_H, img_W / pred_W)
                pred_heatmap_resized = zoom(pred_heatmap, zoom_factor, order=1)
            else:
                pred_heatmap_resized = pred_heatmap
            
            # Normalize heatmap
            if pred_heatmap_resized.max() > pred_heatmap_resized.min():
                heatmap_norm = (pred_heatmap_resized - pred_heatmap_resized.min()) / \
                              (pred_heatmap_resized.max() - pred_heatmap_resized.min())
            else:
                heatmap_norm = pred_heatmap_resized
            
            # Create visualization
            ax = axes[plot_idx]
            ax.imshow(img_normalized)
            
            # Overlay heatmap
            heatmap_colored = cmap(heatmap_norm)[:, :, :3]
            alpha_mask = heatmap_norm * alpha
            
            img_with_heatmap = img_normalized.copy()
            for c in range(3):
                img_with_heatmap[:, :, c] = (1 - alpha_mask) * img_normalized[:, :, c] + \
                                          alpha_mask * heatmap_colored[:, :, c]
            
            ax.imshow(img_with_heatmap)
            
            # Add center point
            ax.plot(center_x_scaled, center_y_scaled, 'r+', markersize=12, markeredgewidth=3)
            ax.plot(center_x_scaled, center_y_scaled, 'w+', markersize=10, markeredgewidth=2)
            
            ax.set_title(f'Batch {batch_idx}, Frame {frame_idx}\n' + 
                        f'Gaze: ({center_x_orig:.1f}, {center_y_orig:.1f}), Conf: {confidence:.2e}')
            ax.axis('off')
            
            plot_idx += 1
        
        all_gaze_centers.append(batch_centers)
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved temporal gaze visualization to {save_path}")
    
    plt.show()
    
    # Print gaze tracking summary
    print(f"\n=== GAZE TRACKING SUMMARY ===")
    for batch_idx, batch_centers in enumerate(all_gaze_centers):
        print(f"Batch {batch_idx}:")
        for frame_data in batch_centers:
            frame = frame_data['frame']
            cx, cy = frame_data['center_original']
            conf = frame_data['confidence']
            print(f"   Frame {frame}: gaze at ({cx:.1f}, {cy:.1f}), confidence: {conf:.2e}")
        
        # Calculate gaze movement
        if len(batch_centers) > 1:
            movements = []
            for i in range(1, len(batch_centers)):
                prev_center = batch_centers[i-1]['center_original']
                curr_center = batch_centers[i]['center_original']
                movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                movements.append(movement)
            
            avg_movement = np.mean(movements)
            max_movement = np.max(movements)
            print(f"   Movement: avg={avg_movement:.1f}, max={max_movement:.1f} pixels")
    
    return all_gaze_centers

def analyze_and_visualize_temporal_gaze(inputs, preds):
    """
    Main function to analyze and visualize temporal gaze predictions.
    """
    print("🔍 Analyzing gaze predictions with temporal interpretation...")
    
    # First, quick analysis
    temporal_interpretation, centers, max_values = analyze_channels_as_temporal(preds)
    
    if temporal_interpretation:
        print("\n✅ Proceeding with temporal visualization...")
        gaze_centers = visualize_temporal_gaze_predictions(
            inputs, preds, 
            save_path="temporal_gaze_predictions.png",
            alpha=0.6,
            max_frames_to_show=8
        )
        return gaze_centers
    else:
        return False 

