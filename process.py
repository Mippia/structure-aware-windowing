import os
import json
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def create_mel_spectrogram(audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """Create mel-spectrogram from audio"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def create_cqt_spectrogram(audio, sr=22050, hop_length=512, n_bins=84):
    """Create CQT spectrogram from audio"""
    cqt = librosa.cqt(y=audio, sr=sr, hop_length=hop_length, n_bins=n_bins)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db

def save_spectrogram(spec, output_path):
    """Save spectrogram as numpy array"""
    np.save(output_path, spec)

def find_best_downbeat(target_time, downbeats, used_downbeats=None, tolerance=0.5):
    """Find the closest unused downbeat to target time"""
    if not downbeats:
        return None
    
    if used_downbeats is None:
        used_downbeats = set()
    
    # Filter out already used downbeats
    available_downbeats = [db for db in downbeats if db not in used_downbeats]
    
    if not available_downbeats:
        return None
    
    distances = [abs(db - target_time) for db in available_downbeats]
    min_idx = np.argmin(distances)
    
    if distances[min_idx] <= tolerance:
        return available_downbeats[min_idx]
    return None

def get_two_bar_segments(data, max_segments=None):
    """Extract 2-bar segments from audio based on BPM and available timing data"""
    bpm = data.get('bpm')
    downbeats = data.get('downbeats', [])
    beats = data.get('beats', [])
    segments = data.get('segments', [])
    
    if not bpm:
        return []
    
    # Calculate 2-bar duration in seconds
    two_bar_duration = (60.0 / bpm) * 8
    
    initial_points = []
    used_downbeats = set()  # Track used downbeats to avoid duplicates
    
    # Priority 1: Use segments start points (main reference points)
    segment_starts = [seg['start'] for seg in segments if seg['start'] > 0.1]
    
    if segment_starts:
        # If we have downbeats, find best matching downbeats for segment starts
        if downbeats:
            for start_time in segment_starts:
                best_downbeat = find_best_downbeat(start_time, downbeats, used_downbeats)
                if best_downbeat is not None:
                    initial_points.append(best_downbeat)
                    used_downbeats.add(best_downbeat)
                else:
                    initial_points.append(start_time)  # Use segment start if no close downbeat
        else:
            # No downbeats, use segment starts directly
            initial_points.extend(segment_starts)
        initial_points = [initial_points[0]] # Actually, it is good to use Quantized segments with bpm and boundaries.. 
    
    # Priority 2: If no good segments, use downbeats
    elif downbeats:
        initial_points.append(downbeats[0])
    
    # Priority 3: If no segments and no downbeats, use beats
    elif beats:
        initial_points.append(beats[0])
    
    # Remove duplicates
    initial_points = sorted(list(set(initial_points)))

    
    # Expand around each initial point (both forward and backward)
    all_start_points = []
    audio_duration = max(beats[-1] + two_bar_duration if beats else 30.0, 30.0)
    
    for initial_point in initial_points:
        # Add the initial point itself
        all_start_points.append(initial_point)
        
        # Expand backward
        backward_time = initial_point - two_bar_duration
        while backward_time >= 0:
            all_start_points.append(backward_time)
            backward_time -= two_bar_duration
        
        # Expand forward
        forward_time = initial_point + two_bar_duration
        while forward_time + two_bar_duration <= audio_duration:
            all_start_points.append(forward_time)
            forward_time += two_bar_duration
    
    # Remove duplicates and sort
    all_start_points = sorted(list(set(all_start_points)))
    
    # If still not enough points, fill with evenly spaced segments
    if len(all_start_points) < 3:  # Minimum threshold instead of num_segments
        max_possible_segments = int(audio_duration / two_bar_duration)
        
        for i in range(max_possible_segments):
            segment_start = i * two_bar_duration
            if segment_start not in all_start_points:
                all_start_points.append(segment_start)
    
    # Final sort and apply max_segments limit if specified
    all_start_points = sorted(list(set(all_start_points)))
    if max_segments is not None:
        all_start_points = all_start_points[:max_segments]
    
    # Return segments with duration
    segments_info = []
    for start in all_start_points:
        segments_info.append({
            'start': start,
            'duration': two_bar_duration
        })
    
    return segments_info

def preprocess_method1_30sec(audio_file, data, output_dir):
    """Method 1: 30-second mel-spectrogram and CQT (fixed 30s)"""
    genre = data['genre']
    filename = Path(audio_file).stem
    
    # Load audio and ensure exactly 30 seconds
    y, sr = librosa.load(audio_file, sr=22050)
    target_samples = int(30.0 * sr)  # Exactly 30 seconds
    
    if len(y) < target_samples:
        # Pad with zeros if shorter than 30s
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    elif len(y) > target_samples:
        # Crop to exactly 30s (take from beginning)
        y = y[:target_samples]
    
    # Create spectrograms
    mel_spec = create_mel_spectrogram(y, sr)
    cqt_spec = create_cqt_spectrogram(y, sr)
    
    # Save
    output_path = os.path.join(output_dir, "method1_30sec", genre)
    os.makedirs(output_path, exist_ok=True)
    
    # Save the 30s wav file as well
    wav_path = os.path.join(output_path, f"{filename}.wav")
    sf.write(wav_path, y, sr)
    
    mel_path = os.path.join(output_path, f"{filename}_mel.npy")
    cqt_path = os.path.join(output_path, f"{filename}_cqt.npy")
    
    save_spectrogram(mel_spec, mel_path)
    save_spectrogram(cqt_spec, cqt_path)
    
    return [wav_path, mel_path, cqt_path]

def preprocess_method2_5sec(audio_file, data, output_dir):
    """Method 2: 6 segments of 5 seconds each"""
    genre = data['genre']
    filename = Path(audio_file).stem
    
    # Load full audio
    y, sr = librosa.load(audio_file, sr=22050)
    
    output_path = os.path.join(output_dir, "method2_5sec", genre)
    os.makedirs(output_path, exist_ok=True)
    
    saved_files = []
    
    # Create 6 segments of 5 seconds
    for i in range(6):
        start_time = i * 5.0
        end_time = start_time + 5.0

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # 원본 오디오 범위 내에서 슬라이싱
        segment = y[start_sample:end_sample]

        # 만약 segment 길이가 5초보다 짧으면 padding
        expected_len = int(5.0 * sr)
        if len(segment) < expected_len:
            segment = np.pad(segment, (0, expected_len - len(segment)))

        # Save wav
        wav_path = os.path.join(output_path, f"{filename}_seg{i:02d}.wav")
        sf.write(wav_path, segment, sr)

        # Create and save spectrograms
        mel_spec = create_mel_spectrogram(segment, sr)
        cqt_spec = create_cqt_spectrogram(segment, sr)

        mel_path = os.path.join(output_path, f"{filename}_seg{i:02d}_mel.npy")
        cqt_path = os.path.join(output_path, f"{filename}_seg{i:02d}_cqt.npy")

        save_spectrogram(mel_spec, mel_path)
        save_spectrogram(cqt_spec, cqt_path)

        saved_files.extend([wav_path, mel_path, cqt_path])

    return saved_files


def preprocess_method3_2bar(audio_file, data, output_dir):
    """Method 3: All possible 2-bar segments"""
    genre = data['genre']
    filename = Path(audio_file).stem
    
    # Get all possible 2-bar segments (no limit)
    segments_info = get_two_bar_segments(data)
    
    if not segments_info:
        print(f"Could not extract 2-bar segments from {filename}")
        return []
    
    # Load full audio
    y, sr = librosa.load(audio_file, sr=22050)
    
    output_path = os.path.join(output_dir, "method3_2bar", genre)
    os.makedirs(output_path, exist_ok=True)
    
    saved_files = []
    
    print(f"Extracting {len(segments_info)} 2-bar segments from {filename}")
    
    for i, seg_info in enumerate(segments_info):
        start_time = seg_info['start']
        duration = seg_info['duration']
        
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        
        if end_sample <= len(y):
            segment = y[start_sample:end_sample]
            
            # Save wav
            wav_path = os.path.join(output_path, f"{filename}_2bar{i:02d}.wav")
            sf.write(wav_path, segment, sr)
            
            # Create and save spectrograms
            mel_spec = create_mel_spectrogram(segment, sr)
            cqt_spec = create_cqt_spectrogram(segment, sr)
            
            mel_path = os.path.join(output_path, f"{filename}_2bar{i:02d}_mel.npy")
            cqt_path = os.path.join(output_path, f"{filename}_2bar{i:02d}_cqt.npy")
            
            save_spectrogram(mel_spec, mel_path)
            save_spectrogram(cqt_spec, cqt_path)
            
            saved_files.extend([wav_path, mel_path, cqt_path])
    
    return saved_files

def process_gtzan_for_classification(analysis_dir="gtzan_analysis", output_dir="gtzan_preprocessed"):
    """Process GTZAN dataset for music classification"""
    
    print("Starting GTZAN preprocessing for music classification...")
    
    # Create output directories
    for method in ["method1_30sec", "method2_5sec", "method3_2bar"]:
        for genre in ['blues', 'classical', 'country', 'disco', 'hiphop', 
                     'jazz', 'metal', 'pop', 'reggae', 'rock']:
            os.makedirs(os.path.join(output_dir, method, genre), exist_ok=True)
    
    stats = {
        'method1': 0,
        'method2': 0,
        'method3': 0,
        'errors': 0
    }
    
    # Process each genre
    for genre in ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']:
        
        genre_path = os.path.join(analysis_dir, genre)
        if not os.path.exists(genre_path):
            continue
        
        json_files = [f for f in os.listdir(genre_path) if f.endswith('.json')]
        print(f"Processing {genre}: {len(json_files)} files")
        
        for json_file in tqdm(json_files, desc=f"Processing {genre}"):
            try:
                json_path = os.path.join(genre_path, json_file)
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                audio_file = data.get('original_path') or data.get('path')
                if not audio_file or not os.path.exists(audio_file):
                    print(f"Audio file not found for {json_file}")
                    continue
                
                # Method 1: 30-second mel-spectrogram
                try:
                    preprocess_method1_30sec(audio_file, data, output_dir)
                    stats['method1'] += 1
                except Exception as e:
                    print(f"Error in method1 for {json_file}: {e}")
                
                # Method 2: 5-second segments
                try:
                    preprocess_method2_5sec(audio_file, data, output_dir)
                    stats['method2'] += 1
                except Exception as e:
                    print(f"Error in method2 for {json_file}: {e}")
                
                # Method 3: 2-bar segments
                try:
                    preprocess_method3_2bar(audio_file, data, output_dir)
                    stats['method3'] += 1
                except Exception as e:
                    print(f"Error in method3 for {json_file}: {e}")
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                stats['errors'] += 1
    
    print(f"\nPreprocessing complete!")
    print(f"Method 1 (30sec): {stats['method1']} files")
    print(f"Method 2 (5sec): {stats['method2']} files")  
    print(f"Method 3 (2bar): {stats['method3']} files")
    print(f"Errors: {stats['errors']} files")

if __name__ == "__main__":
    import sys
    
    analysis_dir = sys.argv[1] if len(sys.argv) > 1 else "gtzan_analysis"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "gtzan_preprocessed"
    
    process_gtzan_for_classification(analysis_dir, output_dir)