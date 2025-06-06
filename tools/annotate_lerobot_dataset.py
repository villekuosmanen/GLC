#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch  # import torch and sklearn at the very beginning to avoid possible incompatible errors
import sklearn

"""Multi-view test a video classification model."""
import numpy as np
import torch
import subprocess
import tempfile
import os
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestGazeMeter
from slowfast.datasets.gaze_exporter import export_gaze_annotations

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from test_gaze_net import test
from train_gaze_net import train

logger = logging.get_logger(__name__)


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'csv=p=0', 
            '-show_entries', 'format=duration', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Failed to get duration for {video_path}: {e}")
        return None


def create_video_clips(dataset, temp_dir, clip_duration=5):
    """
    Create 5-second clips from LeRobotDataset videos.
    
    Args:
        dataset (LeRobotDataset): The dataset containing videos
        temp_dir (Path): Temporary directory to save clips
        clip_duration (int): Duration of each clip in seconds
        
    Returns:
        list: List of paths to created video clips
    """
    video_file_paths = []
    
    logger.info(f"Creating {clip_duration}-second clips from {dataset.num_episodes} episodes")
    logger.info(f"Camera keys: {dataset.meta.camera_keys}")
    
    for episode_idx in range(dataset.num_episodes):
        logger.info(f"Processing episode {episode_idx}/{dataset.num_episodes}")
        
        for camera_key in dataset.meta.camera_keys:
            # Get the original video path
            original_video_path = dataset.root / dataset.meta.get_video_file_path(episode_idx, camera_key)
            
            if not original_video_path.exists():
                logger.warning(f"Video not found: {original_video_path}")
                continue
            
            # Get video duration
            duration = get_video_duration(original_video_path)
            if duration is None:
                logger.warning(f"Skipping {original_video_path} - could not determine duration")
                continue
            
            # Calculate number of clips needed
            num_clips = int(np.ceil(duration / clip_duration))
            
            # Extract camera name from camera_key (e.g., "observation.images.front" -> "front")
            camera_name = camera_key.split('.')[-1] if '.' in camera_key else camera_key
            
            # Create clips
            for clip_idx in range(num_clips):
                start_time = clip_idx * clip_duration
                end_time = min((clip_idx + 1) * clip_duration, duration)
                
                # Skip if clip would be too short (less than 1 second)
                if end_time - start_time < 1.0:
                    logger.info(f"Skipping short clip ({end_time - start_time:.1f}s) for episode {episode_idx}")
                    continue
                
                # Create output filename in the expected format
                clip_filename = f"episode_{episode_idx:06d}_{camera_name}_t{start_time:.0f}_t{end_time:.0f}.mp4"
                clip_path = temp_dir / clip_filename
                
                # Use ffmpeg to create the clip
                cmd = [
                    'ffmpeg', '-y',  # -y to overwrite existing files
                    '-i', str(original_video_path),
                    '-ss', str(start_time),  # Start time
                    '-t', str(end_time - start_time),  # Duration
                    '-c', 'copy',  # Copy streams without re-encoding (faster)
                    '-avoid_negative_ts', 'make_zero',  # Handle timestamp issues
                    str(clip_path)
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    video_file_paths.append(str(clip_path))
                    logger.debug(f"Created clip: {clip_filename}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create clip {clip_filename}: {e}")
                    logger.error(f"FFmpeg stderr: {e.stderr}")
                    continue
    
    logger.info(f"Created {len(video_file_paths)} video clips")
    return video_file_paths


def test(cfg, repo_id):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        repo_id (str): Repository ID for the LeRobotDataset
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Load LeRobotDataset and get video paths
    logger.info(f"Loading LeRobotDataset: {repo_id}")
    dataset = LeRobotDataset(repo_id)
    
    logger.info(f"Dataset info:")
    logger.info(f"  Episodes: {dataset.num_episodes}")
    logger.info(f"  Total frames: {dataset.num_frames}")
    logger.info(f"  FPS: {dataset.fps}")
    logger.info(f"  Camera keys: {dataset.meta.camera_keys}")
    logger.info(f"  Video keys: {dataset.meta.video_keys}")
    
    # Create temporary directory for video clips
    temp_dir = Path(tempfile.mkdtemp(prefix="lerobot_gaze_clips_"))
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create 5-second video clips
        video_file_paths = create_video_clips(dataset, temp_dir, clip_duration=5)
        
        if not video_file_paths:
            logger.error("No video clips were created. Cannot proceed with testing.")
            return
        
        logger.info(f"Created {len(video_file_paths)} video clips for gaze testing")
        
        # Print config.
        logger.info("Test with config:")
        logger.info(cfg)

        # Build the video model and print model statistics.
        model = build_model(cfg)
        if du.is_master_proc() and cfg.LOG_MODEL_INFO:
            misc.log_model_info(model, cfg, use_train_input=False)

        cu.load_test_checkpoint(cfg, model)

        # Create video testing loaders using the clipped videos
        # Note: You may need to modify your loader to accept a list of video paths
        # instead of reading from a CSV file
        test_loader = loader.construct_loader(cfg, "test", video_files=video_file_paths)
        logger.info("Testing model for {} iterations".format(len(test_loader)))

        # Calculate image dimensions based on dataset info
        # Assuming standard dimensions, but you may need to adjust based on your dataset
        json_path = export_gaze_annotations(
            test_loader=test_loader,
            model=model,
            output_file=f"{dataset.root}/meta/gaze_annotations.json",
            image_dims=(640 / 7.5, 480 / 7.5),
        )
        
        # Add gaze annotations to the dataset
        if json_path and Path(json_path).exists():
            logger.info(f"Loading gaze annotations from {json_path}")
            try:
                import json
                with open(json_path, 'r') as f:
                    gaze_data = json.load(f)
                
                # Extract annotations (skip metadata if present)
                annotations = gaze_data.get('annotations', gaze_data)
                
                # Add to dataset
                success = dataset.add_gaze_annotations(annotations)
                if success:
                    logger.info("Successfully added gaze annotations to dataset")
                else:
                    logger.error("Failed to add gaze annotations to dataset")
                    
            except Exception as e:
                logger.error(f"Error processing gaze annotations: {e}")
        
        logger.info("Testing finished!")
        
    finally:
        # Clean up temporary directory
        import shutil
        if temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print(args)
    
    repo_id = args.repo_id
    
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    # Get repo_id from command line arguments
    # You'll need to add this to your argument parser
    

    # Launch LeRobot job
    test(cfg, repo_id)

if __name__ == "__main__":
    main()

# Example usage:
# python lerobot_gaze_test.py --cfg configs/your_config.yaml --repo_id "your_dataset_name"