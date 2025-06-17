import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import json
import os
from pathlib import Path

class SimpleSpectrogramVisualizer:
    def __init__(self):
        pass
    
    def create_mel_spectrogram(self, audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def create_cqt_spectrogram(self, audio, sr=22050, hop_length=512, n_bins=84):
        cqt = librosa.cqt(y=audio, sr=sr, hop_length=hop_length, n_bins=n_bins)
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db
    
    def load_json_data(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def plot_segmentation_comparison(self, json_path, output_dir="visualization_output"):
        data = self.load_json_data(json_path)
        
        # Load original audio file
        original_path = data['original_path']
        if not os.path.exists(original_path):
            # Try alternative path
            original_path = data.get('path', '')
            if not os.path.exists(original_path):
                print(f"Audio file not found: {original_path}")
                return
        
        y, sr = librosa.load(original_path, sr=22050)
        
        # Ensure exactly 30 seconds
        target_samples = int(30.0 * sr)
        if len(y) > target_samples:
            y = y[:target_samples]
        elif len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        
        # Get all timing info
        segments = data.get('segments', [])
        downbeats = data.get('downbeats', [])
        beats = data.get('beats', [])
        
        # Filter out very short segments
        valid_segments = [seg for seg in segments if seg['end'] - seg['start'] > 1.0]
        
        # Get 2-bar segments for the entire song, but align with segment structure
        bar_segments = []
        if len(downbeats) > 0:
            # Find a good starting downbeat based on segment structure (excluding first segment)
            start_downbeat_idx = 0  # Default to first downbeat
            
            if len(valid_segments) > 1:
                # Use the 2nd or 3rd segment start to find a better aligned downbeat
                target_segment = valid_segments[1] if len(valid_segments) > 1 else valid_segments[0]
                target_time = target_segment['start']
                
                # Find the closest downbeat to this segment start
                min_distance = float('inf')
                for i, downbeat in enumerate(downbeats):
                    distance = abs(downbeat - target_time)
                    if distance < min_distance:
                        min_distance = distance
                        start_downbeat_idx = i
            
            # Now create 2-bar segments for the entire song starting from this aligned point
            current_idx = start_downbeat_idx
            
            while current_idx < len(downbeats):
                start_downbeat = downbeats[current_idx]
                
                # Find end point (2 downbeats later)
                if current_idx + 2 < len(downbeats):
                    end_downbeat = downbeats[current_idx + 2]
                elif current_idx + 1 < len(downbeats):
                    # Only 1 more downbeat, estimate 2-bar duration
                    next_downbeat = downbeats[current_idx + 1]
                    bar_duration = next_downbeat - start_downbeat
                    end_downbeat = min(start_downbeat + (bar_duration * 2), 30.0)
                else:
                    # Last downbeat, create reasonable segment
                    end_downbeat = min(start_downbeat + 4.0, 30.0)
                
                # Add segment if valid
                if end_downbeat > start_downbeat and start_downbeat < 30.0:
                    bar_segments.append({'start': start_downbeat, 'end': min(end_downbeat, 30.0)})
                
                # Move to next 2-bar segment
                current_idx += 2
            
            # Also go backwards from the starting point to cover the beginning of the song
            current_idx = start_downbeat_idx - 2
            while current_idx >= 0:
                start_downbeat = downbeats[current_idx]
                
                if current_idx + 2 < len(downbeats):
                    end_downbeat = downbeats[current_idx + 2]
                else:
                    # This shouldn't happen in backward direction, but just in case
                    end_downbeat = min(start_downbeat + 4.0, 30.0)
                
                # Add segment if valid and doesn't overlap
                if end_downbeat > start_downbeat and start_downbeat >= 0:
                    bar_segments.insert(0, {'start': max(start_downbeat, 0.0), 'end': end_downbeat})
                
                current_idx -= 2
            
            # Remove overlapping segments and sort
            bar_segments = sorted(bar_segments, key=lambda x: x['start'])
            
            # Remove duplicates and overlaps
            cleaned_segments = []
            for seg in bar_segments:
                if not cleaned_segments or seg['start'] >= cleaned_segments[-1]['end']:
                    cleaned_segments.append(seg)
            
            bar_segments = cleaned_segments
        
        # Extract filename for saving
        filename = os.path.basename(json_path)
        
        # Create MEL figure (comparison)
        self.create_comparison_plot(y, sr, valid_segments, bar_segments, downbeats, 'mel', output_dir, filename)
        
        # Create CQT figure (comparison)
        self.create_comparison_plot(y, sr, valid_segments, bar_segments, downbeats, 'cqt', output_dir, filename)
        
        # Create single MEL spectrogram
        self.create_single_spectrogram(y, sr, 'mel', output_dir, filename)
        
        # Create single CQT spectrogram
        self.create_single_spectrogram(y, sr, 'cqt', output_dir, filename)
    
    def create_comparison_plot(self, y, sr, segments, bar_segments, downbeats, spec_type, output_dir, filename):
        fig, axes = plt.subplots(4, 1, figsize=(20, 16))  # 4 rows now
        
        # Create the base 30-second spectrogram (same for all three)
        if spec_type == 'mel':
            base_spec = self.create_mel_spectrogram(y, sr)
            cmap = 'magma'
        else:
            base_spec = self.create_cqt_spectrogram(y, sr)
            cmap = 'viridis'
        
        time_to_frame_ratio = base_spec.shape[1] / 30.0
        
        # Define colors for segment labels
        segment_colors = {
            'intro': 'lightblue',
            'verse': 'lightgreen', 
            'chorus': 'orange',
            'bridge': 'lightpink',
            'outro': 'lightyellow',
            'instrumental': 'lightgray'
        }
        
        # Method 1: 30-second (no segmentation)
        ax = axes[0]
        im = ax.imshow(base_spec, aspect='auto', origin='lower', cmap=cmap)
        ax.set_title(f'30-second (No Segmentation)', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Red border for entire spectrogram
        rect = patches.Rectangle((0, 0), base_spec.shape[1]-1, base_spec.shape[0]-1, 
                               linewidth=8, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add segment info boxes above spectrogram
        self.add_segment_boxes(ax, segments, time_to_frame_ratio, base_spec.shape[0], segment_colors, position='above')
        
        # Method 2: 5-second segments
        ax = axes[1]
        im = ax.imshow(base_spec, aspect='auto', origin='lower', cmap=cmap)
        ax.set_title(f'5-second Fixed Segments (6 segments)', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Blue borders for each 5-second segment
        width_per_5sec = base_spec.shape[1] / 6
        for i in range(6):
            start_x = i * width_per_5sec
            width = width_per_5sec
            height = base_spec.shape[0]
            
            rect = patches.Rectangle((start_x, 0), width-1, height-1, 
                                   linewidth=8, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
        
        # Add segment info boxes above spectrogram
        self.add_segment_boxes(ax, segments, time_to_frame_ratio, base_spec.shape[0], segment_colors, position='above')
        
        # Method 3: 2-bar adaptive segments (same color - green)
        ax = axes[2]
        im = ax.imshow(base_spec, aspect='auto', origin='lower', cmap=cmap)
        ax.set_title(f'2-bar Adaptive Segments ({len(bar_segments)} segments)', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Same green color for all 2-bar segments
        for i, bar_seg in enumerate(bar_segments):
            start_time = bar_seg['start']
            end_time = bar_seg['end']
            
            start_x = start_time * time_to_frame_ratio
            end_x = end_time * time_to_frame_ratio
            width = end_x - start_x
            height = base_spec.shape[0]
            
            rect = patches.Rectangle((start_x, 0), width-1, height-1, 
                                   linewidth=8, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        
        # Add segment info boxes above spectrogram
        self.add_segment_boxes(ax, segments, time_to_frame_ratio, base_spec.shape[0], segment_colors, position='above')
        
        # Method 4: Downbeats visualization (new bottom row)
        ax = axes[3]
        ax.set_xlim(0, base_spec.shape[1])
        ax.set_ylim(0, 1)
        ax.set_title(f'Downbeats Pattern', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw downbeat markers
        for downbeat in downbeats:
            if 0 < downbeat < 30.0:
                x_pos = downbeat * time_to_frame_ratio
                ax.axvline(x=x_pos, color='red', linewidth=4, alpha=0.8)
                # Add small red rectangles for better visibility
                rect = patches.Rectangle((x_pos-2, 0.2), 4, 0.6, 
                                       facecolor='red', edgecolor='darkred', alpha=0.8)
                ax.add_patch(rect)
        
        # Add time markers
        for t in range(0, 31, 5):
            x_pos = t * time_to_frame_ratio
            ax.text(x_pos, 0.1, f'{t}s', ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_facecolor('lightgray')
        
        plt.tight_layout()
        
        # Save with filename in the name
        os.makedirs(output_dir, exist_ok=True)
        safe_filename = filename.replace('.json', '').replace('/', '_').replace('\\', '_')
        plt.savefig(f"{output_dir}/{safe_filename}_{spec_type.upper()}.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/{safe_filename}_{spec_type.upper()}.pdf", 
                   bbox_inches='tight')
    def create_single_spectrogram(self, y, sr, spec_type, output_dir, filename):
        """Create a single MEL or CQT spectrogram without segmentation analysis"""
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        
        # Create spectrogram
        if spec_type == 'mel':
            spec = self.create_mel_spectrogram(y, sr)
            cmap = 'magma'
            title = 'MEL Spectrogram'
        else:
            spec = self.create_cqt_spectrogram(y, sr)
            cmap = 'viridis'
            title = 'CQT Spectrogram'
        
        # Plot
        im = ax.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        safe_filename = filename.replace('.json', '').replace('/', '_').replace('\\', '_')
        plt.savefig(f"{output_dir}/{safe_filename}_{spec_type.upper()}_single.png", 
                   dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/{safe_filename}_{spec_type.upper()}_single.pdf", 
                   bbox_inches='tight')
        plt.close()
    
    def add_segment_boxes(self, ax, segments, time_to_frame_ratio, spec_height, segment_colors, position='above'):
        """Add colored boxes showing segment information"""
        
        if position == 'above':
            box_y = spec_height + 5  # Above the spectrogram
            box_height = 15
        else:
            box_y = -20  # Below the spectrogram
            box_height = 15
        
        for seg in segments:
            if seg['end'] - seg['start'] > 0.5:  # Only show segments longer than 0.5 seconds
                start_x = seg['start'] * time_to_frame_ratio
                end_x = seg['end'] * time_to_frame_ratio
                width = end_x - start_x
                
                label = seg.get('label', 'unknown')
                color = segment_colors.get(label, 'lightgray')
                
                # Add colored rectangle
                rect = patches.Rectangle((start_x, box_y), width, box_height, 
                                       facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
                ax.add_patch(rect)
                
                # Add text label if segment is wide enough
                if width > 20:  # Only add text if box is wide enough
                    text_x = start_x + width / 2
                    text_y = box_y + box_height / 2
                    ax.text(text_x, text_y, label, ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='black')
        
        # Extend the plot limits to show the boxes
        current_ylim = ax.get_ylim()
        if position == 'above':
            ax.set_ylim(current_ylim[0], max(current_ylim[1], box_y + box_height + 5))
        else:
            ax.set_ylim(min(current_ylim[0], box_y - 5), current_ylim[1])

def main():
    # Add this at the beginning to prevent memory warnings
    plt.rcParams['figure.max_open_warning'] = 0
    
    visualizer = SimpleSpectrogramVisualizer()
    
    # Process all available JSON files
    base_dir = "/ssd_data/gsh/Segment_importance/gtzan_analysis"
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    all_json_files = []
    
    # Collect all JSON files
    for genre in genres:
        genre_dir = os.path.join(base_dir, genre)
        if os.path.exists(genre_dir):
            for i in range(1, 11):  # 00001 to 00010 for each genre
                json_file = os.path.join(genre_dir, f"{genre}.{i:05d}.json")
                if os.path.exists(json_file):
                    all_json_files.append(json_file)
    
    print(f"Found {len(all_json_files)} JSON files to process")
    all_json_files.reverse()  # Reverse to process in reverse order
    
    # Process each file
    success_count = 0
    fail_count = 0
    
    for i, json_path in enumerate(all_json_files, 1):
        try:
            print(f"[{i}/{len(all_json_files)}] Processing: {os.path.basename(json_path)}")
            visualizer.plot_segmentation_comparison(json_path)
            success_count += 1
            print(f"  ✓ Success")
            
            # Force garbage collection every 10 files to prevent memory issues
            if i % 10 == 0:
                plt.close('all')  # Close any remaining figures
                import gc
                gc.collect()
                print(f"  Memory cleanup at {i} files")
                
        except Exception as e:
            fail_count += 1
            print(f"  ✗ Failed: {e}")
            plt.close('all')  # Close figures even on failure
    
    # Final cleanup
    plt.close('all')
    
    print(f"\nBatch processing complete!")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total files generated: {success_count * 4} (Comparison MEL + Comparison CQT + Single MEL + Single CQT)")

if __name__ == "__main__":
    main()