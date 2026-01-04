import pandas as pd
import numpy as np
from scipy.fftpack import fft

def get_fft_features(df, window_size=50):
    """
    Converts raw XYZ data into Frequency Domain features.
    """
    features = []
    labels = []
    
    # 1. Map labels to 'Harvestable' (1) or 'Not' (0)
    harvestable_map = {
        'traffic': 1, 'footsteps': 1, 'construction': 1,
        'idle': 0, 'wind': 0, 'noise': 0, 'animals': 0
    }
    
    # 2. Process windows
    for i in range(0, len(df) - window_size, window_size):
        window = df.iloc[i:i+window_size]
        
        # Calculate Magnitude of acceleration vector
        mag = np.sqrt(window['accel_x']**2 + window['accel_y']**2 + window['accel_z']**2)
        
        # Apply FFT
        y_fft = np.abs(fft(mag.values))
        # Keep only the positive frequency half
        y_fft = y_fft[:window_size//2] 
        
        features.append(y_fft)
        labels.append(harvestable_map[window['label'].iloc[0]])
        
    return np.array(features), np.array(labels)