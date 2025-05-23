#!/usr/bin/env python3
"""
Split the large processed_dual_bearing_data.h5 file into two separate files
for bearing A and bearing B to stay under GitHub file size limits.
"""

import h5py
import numpy as np
import os

def split_bearing_data():
    """Split the processed dual bearing data into separate files for each bearing"""
    
    input_file = 'processed_dual_bearing_data.h5'
    bearing_a_file = 'processed_bearing_a_data.h5'
    bearing_b_file = 'processed_bearing_b_data.h5'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print(f"📂 Loading data from {input_file}...")
    
    # Load the original data
    with h5py.File(input_file, 'r') as h5f:
        # Load shared data
        time_vector = h5f['time_vector'][:]
        freqs = h5f['freqs'][:]
        
        # Load metadata
        sampling_frequency = h5f.attrs['sampling_frequency']
        shaft_speed_rpm = h5f.attrs['shaft_speed_rpm']
        fundamental_frequency = h5f.attrs['fundamental_frequency_hz']
        
        # Load bearing A data
        bearing_a_fft_mags = h5f['bearing_a']['fft_mags'][:]
        bearing_a_fft_phases = h5f['bearing_a']['fft_phases'][:]
        bearing_a_angles = h5f['bearing_a']['sensor_angles'][:]
        bearing_a_rolling_elements = h5f['bearing_a'].attrs['rolling_elements']
        
        # Load bearing B data
        bearing_b_fft_mags = h5f['bearing_b']['fft_mags'][:]
        bearing_b_fft_phases = h5f['bearing_b']['fft_phases'][:]
        bearing_b_angles = h5f['bearing_b']['sensor_angles'][:]
        bearing_b_rolling_elements = h5f['bearing_b'].attrs['rolling_elements']
    
    print(f"✅ Loaded data successfully")
    print(f"   Time vector shape: {time_vector.shape}")
    print(f"   Frequency vector shape: {freqs.shape}")
    print(f"   Bearing A FFT shape: {bearing_a_fft_mags.shape}")
    print(f"   Bearing B FFT shape: {bearing_b_fft_mags.shape}")
    
    # Create bearing A file
    print(f"💾 Creating {bearing_a_file}...")
    with h5py.File(bearing_a_file, 'w') as h5f:
        # Store shared data
        h5f.create_dataset('time_vector', data=time_vector)
        h5f.create_dataset('freqs', data=freqs)
        
        # Store metadata
        h5f.attrs['sampling_frequency'] = sampling_frequency
        h5f.attrs['shaft_speed_rpm'] = shaft_speed_rpm
        h5f.attrs['fundamental_frequency_hz'] = fundamental_frequency
        h5f.attrs['bearing_type'] = 'A'
        
        # Store bearing A data
        bearing_a_group = h5f.create_group('bearing_data')
        bearing_a_group.create_dataset('fft_mags', data=bearing_a_fft_mags)
        bearing_a_group.create_dataset('fft_phases', data=bearing_a_fft_phases)
        bearing_a_group.create_dataset('sensor_angles', data=bearing_a_angles)
        bearing_a_group.attrs['rolling_elements'] = bearing_a_rolling_elements
        bearing_a_group.attrs['num_sensors'] = len(bearing_a_angles)
    
    # Create bearing B file
    print(f"💾 Creating {bearing_b_file}...")
    with h5py.File(bearing_b_file, 'w') as h5f:
        # Store shared data
        h5f.create_dataset('time_vector', data=time_vector)
        h5f.create_dataset('freqs', data=freqs)
        
        # Store metadata
        h5f.attrs['sampling_frequency'] = sampling_frequency
        h5f.attrs['shaft_speed_rpm'] = shaft_speed_rpm
        h5f.attrs['fundamental_frequency_hz'] = fundamental_frequency
        h5f.attrs['bearing_type'] = 'B'
        
        # Store bearing B data
        bearing_b_group = h5f.create_group('bearing_data')
        bearing_b_group.create_dataset('fft_mags', data=bearing_b_fft_mags)
        bearing_b_group.create_dataset('fft_phases', data=bearing_b_fft_phases)
        bearing_b_group.create_dataset('sensor_angles', data=bearing_b_angles)
        bearing_b_group.attrs['rolling_elements'] = bearing_b_rolling_elements
        bearing_b_group.attrs['num_sensors'] = len(bearing_b_angles)
    
    # Check file sizes
    bearing_a_size = os.path.getsize(bearing_a_file) / (1024*1024)  # MB
    bearing_b_size = os.path.getsize(bearing_b_file) / (1024*1024)  # MB
    
    print(f"✅ Successfully created split files:")
    print(f"   📁 {bearing_a_file}: {bearing_a_size:.1f} MB")
    print(f"   📁 {bearing_b_file}: {bearing_b_size:.1f} MB")
    
    if bearing_a_size < 25 and bearing_b_size < 25:
        print(f"🎯 Both files are under 25MB - suitable for GitHub!")
    else:
        print(f"⚠️  Warning: One or both files are still large")
    
    return bearing_a_file, bearing_b_file

if __name__ == "__main__":
    split_bearing_data() 