import os
import subprocess
import json
from pathlib import Path
from tqdm import tqdm
import sys
import madmom
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import numpy as np


def process_with_madmom(audio_file):
    
    try:
        beat_proc = RNNBeatProcessor()
        beat_tracker = DBNBeatTrackingProcessor(fps=100)
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        
        # Process beats and downbeats
        beat_activations = beat_proc(audio_file)
        beats = beat_tracker(beat_activations)
        
        downbeat_activations = downbeat_proc(audio_file)
        downbeats_data = downbeat_tracker(downbeat_activations)
        downbeats = downbeats_data[downbeats_data[:, 1] == 1, 0] if len(downbeats_data.shape) > 1 else []
        
        # Calculate BPM from beats
        bpm = 60.0 / np.median(np.diff(beats)) if len(beats) > 1 else None
        
        return {
            'bpm': float(bpm) if bpm else None,
            'beats': beats.tolist(),
            'downbeats': downbeats.tolist()
        }
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def fix_existing_analysis(output_dir="gtzan_analysis"):
    """Fix existing analysis files with missing data"""
    if not MADMOM_AVAILABLE:
        print("madmom not available. Install with: pip install madmom")
        return
    
    fixed_count = 0
    total_count = 0
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                total_count += 1
                
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Check if needs fixing
                needs_fix = (not data.get('bpm') or 
                           not data.get('beats') or 
                           not data.get('downbeats'))
                
                if needs_fix:
                    # Find audio file
                    audio_name = os.path.splitext(file)[0]
                    genre = os.path.basename(root)
                    
                    audio_file = None
                    for ext in ['.wav', '.mp3', '.au']:
                        path = f"Data/genres_original/{genre}/{audio_name}{ext}"
                        if os.path.exists(path):
                            audio_file = path
                            break
                    
                    if audio_file:
                        print(f"Fixing {file}...")
                        madmom_results = process_with_madmom(audio_file)
                        
                        if madmom_results:
                            if not data.get('bpm'):
                                data['bpm'] = madmom_results['bpm']
                            if not data.get('beats'):
                                data['beats'] = madmom_results['beats']
                            if not data.get('downbeats'):
                                data['downbeats'] = madmom_results['downbeats']
                            
                            data['fixed_with_madmom'] = True
                            
                            with open(json_path, 'w') as f:
                                json.dump(data, f, indent=2)
                            
                            fixed_count += 1
    
    print(f"Fixed {fixed_count}/{total_count} files")

def process_gtzan():
    """Process GTZAN dataset with allin1"""
    gtzan_root = "Data/genres_original"
    
    # Find all audio files
    audio_files = []
    for root, dirs, files in os.walk(gtzan_root):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.au')):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process with allin1
    for i in tqdm(range(0, len(audio_files), 10)):
        batch = audio_files[i:i+10]
        try:
            subprocess.run(["allin1"] + batch, timeout=300)
        except Exception as e:
            print(f"Error in batch: {e}")
    
    # Organize results
    os.makedirs("gtzan_analysis", exist_ok=True)
    
    for audio_file in audio_files:
        audio_path = Path(audio_file)
        json_file = f"./struct/{audio_path.stem}.json"
        
        if os.path.exists(json_file):
            # Extract genre
            genre = None
            for part in audio_path.parts:
                if part in ['blues', 'classical', 'country', 'disco', 'hiphop', 
                           'jazz', 'metal', 'pop', 'reggae', 'rock']:
                    genre = part
                    break
            
            if genre:
                genre_dir = f"gtzan_analysis/{genre}"
                os.makedirs(genre_dir, exist_ok=True)
                
                dest_path = f"{genre_dir}/{audio_path.stem}.json"
                os.rename(json_file, dest_path)
                
                # Add metadata
                with open(dest_path, 'r') as f:
                    data = json.load(f)
                data['original_path'] = audio_file
                data['genre'] = genre
                with open(dest_path, 'w') as f:
                    json.dump(data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fix":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "gtzan_analysis"
        fix_existing_analysis(output_dir)
    else:
        process_gtzan()