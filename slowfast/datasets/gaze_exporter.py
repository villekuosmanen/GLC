import json
import os
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict

from slowfast.utils.utils import frame_softmax

class GazeAnnotationExporter:
    """
    Export gaze predictions to JSON with nested structure:
    video_name -> clip_name -> frame_idx -> {x, y}
    """
    
    def __init__(self, output_path="gaze_annotations.json", image_dims=(256, 256)):
        """
        Args:
            output_path (str): Path to save the JSON file
            image_dims (tuple): Original image dimensions (width, height) for normalization
        """
        self.output_path = output_path
        self.image_width, self.image_height = image_dims
        # Structure: episode_id -> frame_id -> camera_name -> {x, y}
        self.annotations = defaultdict(lambda: defaultdict(dict))
        
    def extract_episode_and_camera_info(self, video_path):
        """
        Extract episode ID and camera name from video path.
        
        Args:
            video_path (str): Path like '/path/to/episode_000006_front_t0_t5.mp4' or 'episode_000003_left_wrist_t0_t5.mp4'
            
        Returns:
            tuple: (episode_id, camera_name)
        """
        path = Path(video_path)
        filename = path.stem  # Remove .mp4 extension
        
        # Split filename to extract parts
        parts = filename.split('_')
        
        episode_id = None
        camera_name = None
        
        # Look for episode pattern like 'episode_000006'
        for i, part in enumerate(parts):
            if part == 'episode' and i < len(parts) - 1:
                try:
                    # Extract episode number (remove leading zeros)
                    episode_num_str = parts[i + 1]
                    episode_id = str(int(episode_num_str))  # Convert to int then back to string to remove leading zeros
                    
                    # Camera name: everything after episode number until we hit time patterns (t0, t5, etc.)
                    camera_parts = []
                    for j in range(i + 2, len(parts)):
                        # Stop if we hit a time pattern like 't0', 't5', 't10', etc.
                        if parts[j].startswith('t') and parts[j][1:].isdigit():
                            break
                        camera_parts.append(parts[j])
                    
                    if camera_parts:
                        camera_name = f"observation.images.{'_'.join(camera_parts)}"
                    break
                except ValueError:
                    continue
        
        # Fallback if pattern not found
        if episode_id is None:
            print(f"⚠️  Warning: Could not extract episode ID from {filename}")
            # Try to extract any number from the filename as fallback
            import re
            numbers = re.findall(r'\d+', filename)
            if numbers:
                episode_id = str(int(numbers[0]))  # Use first number found
            else:
                episode_id = "unknown"
        
        if camera_name is None:
            print(f"⚠️  Warning: Could not extract camera name from {filename}")
            # Try to extract camera name by removing known patterns
            filtered_parts = []
            for part in parts:
                # Skip 'episode', numbers, and time patterns
                if (part != 'episode' and 
                    not part.isdigit() and 
                    not (part.startswith('t') and part[1:].isdigit())):
                    filtered_parts.append(part)
            
            if filtered_parts:
                camera_name = f"observation.images.{'_'.join(filtered_parts)}"
            else:
                camera_name = "observation.images.unknown"
        
        return episode_id, camera_name

    def test_filename_extraction(self):
        """
        Test the filename extraction with various patterns.
        """
        test_cases = [
            "episode_000006_front_t0_t5.mp4",
            "episode_000003_left_wrist_t0_t5.mp4", 
            "episode_000012_right_wrist_t5_t10.mp4",
            "episode_000001_overhead_camera_t0_t5.mp4",
            "episode_000099_wrist_t10_t15.mp4",
            "/path/to/episode_000006_front_t0_t5.mp4"
        ]
        
        print("🧪 Testing filename extraction:")
        for filename in test_cases:
            episode_id, camera_name = self.extract_episode_and_camera_info(filename)
            print(f"   '{filename}' → Episode: {episode_id}, Camera: {camera_name}")
        print()
    
    def calculate_frame_indices(self, meta, frame_idx_in_clip, batch_idx=0):
        """
        Calculate global frame indices from metadata.
        
        Args:
            meta (dict): Metadata containing frame indices
            frame_idx_in_clip (int): Frame index within the 8-frame clip (0-7)
            batch_idx (int): Batch index to get the correct frame indices
            
        Returns:
            int: Global frame index
        """
        if 'index' in meta and hasattr(meta['index'], 'shape'):
            # meta['index'] contains the global frame indices for the 8 sampled frames
            frame_indices = meta['index']
            if isinstance(frame_indices, torch.Tensor):
                frame_indices = frame_indices.cpu().numpy()
            
            # Get the global frame index for this specific frame and batch
            if (batch_idx < frame_indices.shape[0] and 
                frame_idx_in_clip < frame_indices.shape[1]):
                global_frame_idx = frame_indices[batch_idx, frame_idx_in_clip]
                
                # Handle negative indices (bug from earlier)
                if global_frame_idx < 0:
                    print(f"⚠️  Warning: negative frame index {global_frame_idx} detected, using frame_idx_in_clip")
                    return frame_idx_in_clip
                
                return int(global_frame_idx)
            else:
                print(f"⚠️  Warning: batch_idx {batch_idx} or frame_idx {frame_idx_in_clip} out of bounds")
        
        # Fallback: use frame_idx_in_clip
        print(f"⚠️  Warning: using fallback frame index {frame_idx_in_clip}")
        return frame_idx_in_clip
    
    def normalize_gaze_coordinates(self, gaze_x, gaze_y, pred_dims=(64, 64)):
        """
        Normalize gaze coordinates to [0, 1] range based on image dimensions.
        
        Args:
            gaze_x, gaze_y (float): Gaze coordinates in prediction space
            pred_dims (tuple): Dimensions of prediction heatmap (width, height)
            
        Returns:
            tuple: (normalized_x, normalized_y) in [0, 1] range
        """
        pred_width, pred_height = pred_dims
        
        # First normalize to [0, 1] based on prediction dimensions
        norm_x = gaze_x / pred_width
        norm_y = gaze_y / pred_height
        
        # Clamp to [0, 1] range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        
        return norm_x, norm_y
    
    def add_batch_annotations(self, inputs, preds, video_idx, meta):
        """
        Add gaze annotations from a batch to the collection.
        
        Args:
            inputs: Input tensor [batch, channels, temporal, H, W]
            preds: Predictions [batch, 1, 8, H, W] (treating 8 as temporal)
            video_idx: Video indices tensor
            meta: Metadata dict with paths and frame indices
        """
        # Reshape predictions to treat channels as temporal
        if preds.shape[1] == 1 and preds.shape[2] == 8:
            preds_temporal = preds.permute(0, 2, 1, 3, 4)  # [batch, 8, 1, H, W]
        else:
            preds_temporal = preds
        
        # Convert to numpy
        if isinstance(preds_temporal, torch.Tensor):
            preds_temporal = preds_temporal.cpu().numpy()
        
        batch_size = preds_temporal.shape[0]
        temporal_frames = preds_temporal.shape[1]
        pred_height, pred_width = preds_temporal.shape[3], preds_temporal.shape[4]
        
        for batch_idx in range(batch_size):
            # Extract episode and camera info
            video_path = meta['path'][batch_idx]
            episode_id, camera_name = self.extract_episode_and_camera_info(video_path)
            
            print(f"Processing: Episode {episode_id}, Camera {camera_name}")
            
            # Debug: Print frame indices for this batch
            if 'index' in meta:
                frame_indices = meta['index']
                if isinstance(frame_indices, torch.Tensor):
                    frame_indices_np = frame_indices.cpu().numpy()
                else:
                    frame_indices_np = frame_indices
                
                if batch_idx < frame_indices_np.shape[0]:
                    batch_frame_indices = frame_indices_np[batch_idx]
                    print(f"   Frame indices for batch {batch_idx}: {batch_frame_indices}")
            
            # Process each temporal frame
            for temporal_idx in range(temporal_frames):
                # Get prediction heatmap
                heatmap = preds_temporal[batch_idx, temporal_idx, 0, :, :]  # [H, W]
                
                # Find gaze center
                center_y, center_x, confidence = self.find_gaze_center_point(heatmap)
                
                # Normalize coordinates
                norm_x, norm_y = self.normalize_gaze_coordinates(
                    center_x, center_y, 
                    pred_dims=(pred_width, pred_height)
                )
                
                # Calculate global frame index
                global_frame_idx = self.calculate_frame_indices(meta, temporal_idx, batch_idx)
                
                # Handle negative frame indices (from the earlier bug)
                if global_frame_idx < 0:
                    print(f"⚠️  Negative frame index {global_frame_idx} detected for {video_name}")
                    global_frame_idx = temporal_idx
                
                # Create annotation data
                annotation_data = {
                    "x": float(norm_x),
                    "y": float(norm_y)
                }
                
                # Store annotation: episode_id -> frame_id -> camera_name -> {x, y}
                frame_id = str(global_frame_idx)
                
                # Check for duplicate camera data for this episode/frame
                if (episode_id in self.annotations and 
                    frame_id in self.annotations[episode_id] and 
                    camera_name in self.annotations[episode_id][frame_id]):
                    print(f"⚠️  WARNING: Overwriting {camera_name} data for episode {episode_id}, frame {frame_id}")
                    print(f"   Previous: {self.annotations[episode_id][frame_id][camera_name]}")
                    print(f"   New: {annotation_data}")
                
                # Store the annotation
                self.annotations[episode_id][frame_id][camera_name] = annotation_data
                
                print(f"   Episode {episode_id}, Frame {frame_id}, {camera_name}: gaze=({norm_x:.3f}, {norm_y:.3f}), conf={confidence:.2e}")
    
    def find_gaze_center_point(self, heatmap, method='weighted_centroid'):
        """
        Find the center point of gaze prediction from a heatmap.
        """
        if method == 'weighted_centroid':
            y_coords, x_coords = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
            weights = heatmap / (heatmap.sum() + 1e-8)
            center_y = np.sum(y_coords * weights)
            center_x = np.sum(x_coords * weights)
            confidence = heatmap.max()
        else:  # max_point fallback
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            center_y, center_x = max_idx
            confidence = heatmap.max()
        
        return center_y, center_x, confidence
    
    def interpolate_gaze_between_keyframes(self):
        """
        Interpolate gaze coordinates for all frames between sampled keyframes.
        For each episode and camera, fill in missing frames with linear interpolation.
        """
        print("\n🔄 Interpolating gaze data between keyframes...")
        
        for episode_id in self.annotations:
            print(f"   Processing Episode {episode_id}...")
            
            # Get all frames for this episode, sorted numerically
            all_frames = sorted(self.annotations[episode_id].keys(), key=lambda x: int(x))
            
            if len(all_frames) < 2:
                print(f"     ⚠️  Only {len(all_frames)} frame(s), skipping interpolation")
                continue
            
            # Get all unique cameras across all frames in this episode
            all_cameras = set()
            for frame_id in all_frames:
                all_cameras.update(self.annotations[episode_id][frame_id].keys())
            
            print(f"     Cameras found: {sorted(all_cameras)}")
            print(f"     Keyframes: {all_frames}")
            
            # Interpolate for each camera
            for camera_name in all_cameras:
                # Find keyframes that have data for this camera
                keyframes_with_data = []
                for frame_id in all_frames:
                    if camera_name in self.annotations[episode_id][frame_id]:
                        keyframes_with_data.append(int(frame_id))
                
                if len(keyframes_with_data) < 2:
                    print(f"       📷 {camera_name}: Only {len(keyframes_with_data)} keyframe(s), skipping")
                    continue
                
                keyframes_with_data.sort()
                print(f"       📷 {camera_name}: Keyframes with data: {keyframes_with_data}")
                
                # Interpolate between consecutive keyframes
                total_interpolated = 0
                for i in range(len(keyframes_with_data) - 1):
                    start_frame = keyframes_with_data[i]
                    end_frame = keyframes_with_data[i + 1]
                    
                    # Get start and end coordinates
                    start_coords = self.annotations[episode_id][str(start_frame)][camera_name]
                    end_coords = self.annotations[episode_id][str(end_frame)][camera_name]
                    
                    start_x, start_y = start_coords['x'], start_coords['y']
                    end_x, end_y = end_coords['x'], end_coords['y']
                    
                    # Calculate how many frames need interpolation
                    frames_between = end_frame - start_frame - 1
                    
                    if frames_between > 0:
                        print(f"         Interpolating {frames_between} frames between {start_frame} and {end_frame}")
                        
                        # Linear interpolation for each intermediate frame
                        for j in range(1, frames_between + 1):
                            intermediate_frame = start_frame + j
                            
                            # Calculate interpolation factor (0 to 1)
                            alpha = j / (frames_between + 1)
                            
                            # Linear interpolation
                            interp_x = start_x + alpha * (end_x - start_x)
                            interp_y = start_y + alpha * (end_y - start_y)
                            
                            # Ensure the frame entry exists
                            frame_id_str = str(intermediate_frame)
                            if frame_id_str not in self.annotations[episode_id]:
                                self.annotations[episode_id][frame_id_str] = {}
                            
                            # Add interpolated data
                            self.annotations[episode_id][frame_id_str][camera_name] = {
                                "x": float(interp_x),
                                "y": float(interp_y),
                                "interpolated": True  # Mark as interpolated
                            }
                            
                            total_interpolated += 1
                
                print(f"         Added {total_interpolated} interpolated frames")
        
        print("✅ Interpolation complete!")

    def save_annotations(self, include_metadata=True, interpolate=True):
        """
        Save all collected annotations to JSON file.
        
        Args:
            include_metadata (bool): Whether to include metadata in the JSON
            interpolate (bool): Whether to interpolate between keyframes before saving
            
        Returns:
            str: Path to saved file
        """
        # Interpolate missing frames if requested
        if interpolate:
            self.interpolate_gaze_between_keyframes()
        
        # Prepare final structure
        output_data = dict(self.annotations)
        
        # Convert nested defaultdicts to regular dicts and ensure proper sorting
        final_data = {}
        for episode_id in sorted(output_data.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
            final_data[episode_id] = {}
            for frame_id in sorted(output_data[episode_id].keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
                final_data[episode_id][frame_id] = dict(output_data[episode_id][frame_id])
        
        if include_metadata:
            # Add metadata
            metadata = {
                "image_dimensions": {
                    "width": self.image_width,
                    "height": self.image_height
                },
                "coordinate_format": "normalized",
                "coordinate_range": "[0, 1]", 
                "description": "Gaze annotations organized by episode and frame",
                "structure": "episode_id -> frame_id -> camera_name -> {x, y}",
                "interpolation": "Linear interpolation applied between keyframes" if interpolate else "No interpolation applied"
            }
            
            final_output = {
                "metadata": metadata,
                "annotations": final_data
            }
        else:
            final_output = final_data
        
        # Save to file
        with open(self.output_path, 'w') as f:
            json.dump(final_output, f, indent=2, sort_keys=False)  # Don't sort keys to preserve episode/frame order
        
        # Print summary
        total_episodes = len(final_data)
        total_frames = sum(len(frames) for frames in final_data.values())
        total_cameras = set()
        interpolated_frames = 0
        keyframe_count = 0
        
        for episode_data in final_data.values():
            for frame_data in episode_data.values():
                total_cameras.update(frame_data.keys())
                for camera_data in frame_data.values():
                    if camera_data.get('interpolated', False):
                        interpolated_frames += 1
                    else:
                        keyframe_count += 1
        
        print(f"\n📁 Saved gaze annotations to: {self.output_path}")
        print(f"   Episodes: {total_episodes}")
        print(f"   Total frames: {total_frames}")
        print(f"   Keyframes (original): {keyframe_count}")
        print(f"   Interpolated frames: {interpolated_frames}")
        print(f"   Cameras: {sorted(total_cameras)}")
        
        return self.output_path
    
    def print_structure_preview(self):
        """
        Print a preview of the annotation structure.
        """
        print("\n📋 Annotation Structure Preview:")
        
        # Structure: episode_id -> frame_id -> camera_name -> {x, y}
        for episode_id, frames in list(self.annotations.items())[:2]:  # Show first 2 episodes
            print(f"  📼 Episode {episode_id}:")
            # Sort frame IDs numerically for display
            sorted_frames = sorted(frames.items(), key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
            for frame_id, cameras in sorted_frames[:5]:  # Show first 5 frames
                print(f"    🎯 Frame {frame_id}:")
                for camera_name, annotation in cameras.items():
                    x, y = annotation['x'], annotation['y']
                    print(f"      📷 {camera_name}: x={x:.3f}, y={y:.3f}")
            if len(frames) > 5:
                print(f"    ... and {len(frames) - 5} more frames")
        if len(self.annotations) > 2:
            print(f"  ... and {len(self.annotations) - 2} more episodes")

# Usage function for your testing loop
def export_gaze_annotations(test_loader, model, output_file="gaze_annotations.json", image_dims=(256, 256), interpolate=True):
    """
    Main function to export gaze annotations from your test loop.
    
    Args:
        test_loader: Your test data loader
        model: Trained gaze model
        output_file (str): Output JSON file path
        image_dims (tuple): Image dimensions for normalization
        interpolate (bool): Whether to interpolate between keyframes
        
    Returns:
        str: Path to saved JSON file
    """
    print(f"🚀 Starting gaze annotation export...")
    print(f"   Output format: episode_id -> frame_id -> camera_name -> {{x, y}}")
    print(f"   Interpolation: {'Enabled' if interpolate else 'Disabled'}")
    
    # Initialize exporter
    exporter = GazeAnnotationExporter(output_file, image_dims)
    
    model.eval()
    
    with torch.no_grad():
        for cur_iter, (inputs, labels_hm, video_idx, meta) in enumerate(test_loader):
            print(f"\nProcessing batch {cur_iter + 1}/{len(test_loader)}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                if isinstance(inputs, list):
                    inputs = [inp.cuda() for inp in inputs]
                else:
                    inputs = inputs.cuda()
            
            # Get predictions
            preds = model(inputs)
            # Apply your softmax (assuming you have this function)
            preds = frame_softmax(preds, temperature=2)
            
            # Add to annotations
            exporter.add_batch_annotations(inputs, preds, video_idx, meta)
    
    # Save results
    exporter.print_structure_preview()
    output_path = exporter.save_annotations(include_metadata=False, interpolate=interpolate)
    
    return output_path

# Example usage in your test script:
"""
# Export with interpolation (recommended):
output_json = export_gaze_annotations(
    test_loader=test_loader,
    model=model,
    output_file="gaze_annotations_interpolated.json",
    image_dims=(256, 256),
    interpolate=True  # Fill in gaps between keyframes
)

# Export without interpolation (keyframes only):
output_json = export_gaze_annotations(
    test_loader=test_loader,
    model=model,
    output_file="gaze_annotations_keyframes_only.json", 
    image_dims=(256, 256),
    interpolate=False  # Only original keyframes
)
"""