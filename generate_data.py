import numpy as np
import pandas as pd
import sqlite3
import scipy.fft
import os

def generate_raw_signal(signal_type='normal', duration=1.0, fs=1000):
    """
    Generates 1 second of data at 1000Hz.
    Normal: 50Hz sine wave + low noise.
    Faulty: 50Hz sine wave + high noise + random spikes.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Base 50Hz sine wave
    signal = np.sin(2 * np.pi * 50 * t)
    
    if signal_type == 'normal':
        # Low noise
        noise = np.random.normal(0, 0.1, len(t))
        signal += noise
    elif signal_type == 'faulty':
        # High noise
        noise = np.random.normal(0, 0.5, len(t))
        signal += noise
        # Random spikes (bearing impacts)
        num_spikes = np.random.randint(5, 11)
        spike_indices = np.random.randint(0, len(t), num_spikes)
        for idx in spike_indices:
            # Add randomized spikes of significant magnitude
            signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(2.0, 5.0)
            
    return signal.astype(np.float32)

def create_sql_pipeline(db_name='sensor_logs.db', n_samples=1000):
    """
    Generates n_samples (mixed types).
    Saves them into a local SQLite database (sensor_logs.db).
    Reads them back into a Pandas DataFrame.
    """
    # Clean up existing db if we want a fresh run, but usually pipeline appends or creates.
    # The prompt implies a fresh creation for the demo.
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table with BLOB storage for the array
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_type TEXT,
            raw_signal BLOB
        )
    ''')
    
    # Generate batch data
    data_entries = []
    
    for _ in range(n_samples):
        # Randomly choose type (balanced 50/50 approx)
        s_type = np.random.choice(['normal', 'faulty'])
        sig = generate_raw_signal(s_type)
        # Store numpy array as bytes
        data_entries.append((s_type, sig.tobytes()))
        
    cursor.executemany('INSERT INTO sensor_data (signal_type, raw_signal) VALUES (?, ?)', data_entries)
    conn.commit()
    
    # Read back
    query = "SELECT * FROM sensor_data"
    df = pd.read_sql(query, conn)
    
    conn.close()
    
    # Convert BLOB bytes back to numpy array
    # We used astype(np.float32) previously, so we must match that
    df['raw_signal'] = df['raw_signal'].apply(lambda x: np.frombuffer(x, dtype=np.float32))
    
    return df

def apply_fft_processing(signal):
    """
    Uses scipy.fft to convert the raw time-domain signal into frequency-domain features.
    Returns magnitude spectrum (positive frequencies).
    """
    # Basic FFT
    yf = scipy.fft.fft(signal)
    # Magnitude
    mag = np.abs(yf)
    # Take the first half (Nyquist)
    n = len(signal)
    return mag[:n//2]

def get_processed_data():
    """
    Runs the pipeline, applies FFT to all rows, and returns X (reshaped for CNN) and y.
    """
    # Simply remove old DB for clean execution in this script
    if os.path.exists('sensor_logs.db'):
        os.remove('sensor_logs.db')
        
    print("Generating data and running SQL pipeline...")
    df = create_sql_pipeline(n_samples=1000)
    
    print("Applying FFT processing...")
    # Apply FFT to each row
    # Result is a list of arrays
    min_len = 0
    X_features = []
    
    for sig in df['raw_signal']:
        fft_res = apply_fft_processing(sig)
        X_features.append(fft_res)
        
    X = np.array(X_features)
    
    # Create y
    # Map 'normal' -> 0, 'faulty' -> 1
    y = df['signal_type'].map({'normal': 0, 'faulty': 1}).values
    
    # Reshape for CNN: (samples, time_steps, channels)
    # X shape is currently (samples, features)
    # We want (samples, features, 1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    print(f"Processed Data Ready.")
    print(f"X shape: {X.shape}")       # Expect (1000, 500, 1)
    print(f"y shape: {y.shape}")       # Expect (1000,)
    
    return X, y

if __name__ == "__main__":
    X, y = get_processed_data()
    print("Sample X[0] stats:", np.min(X[0]), np.max(X[0]))
    print("Sample y[:10]:", y[:10])
