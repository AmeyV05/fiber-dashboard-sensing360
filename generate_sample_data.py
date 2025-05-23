#!/usr/bin/env python3
"""
Generate Sample Data for Dual Bearing Dashboard

This script creates a lightweight sample dataset for demonstration purposes
when the full dataset is not available (e.g., on Streamlit Cloud).
"""

import numpy as np
import h5py
import os

def generate_sample_data():
    """Generate sample dual bearing data for demonstration"""
    print("🔧 Generating sample dual bearing data...")
    
    # Parameters
    sampling_frequency = 2000  # Hz
    duration = 10  # seconds (much smaller than full dataset)
    shaft_speed_rpm = 450
    fundamental_frequency = shaft_speed_rpm / 60  # 7.5 Hz
    
    # Time vector
    num_samples = int(sampling_frequency * duration)
    time_vector = np.linspace(0, duration, num_samples)
    
    # Frequency vector for FFT
    freqs = np.fft.fftfreq(num_samples, 1/sampling_frequency)
    freqs = freqs[:num_samples//2]  # Only positive frequencies
    
    # Bearing A configuration
    bearing_a_sensors = 13
    bearing_a_angles = np.linspace(67.0, 399.3, bearing_a_sensors)
    bearing_a_rolling_elements = 19
    
    # Bearing B configuration  
    bearing_b_sensors = 13
    bearing_b_angles = np.linspace(282.0, -50.3, bearing_b_sensors)
    bearing_b_rolling_elements = 21
    
    print(f"📊 Sample parameters:")
    print(f"   Duration: {duration} seconds")
    print(f"   Sampling rate: {sampling_frequency} Hz")
    print(f"   Samples: {num_samples}")
    print(f"   Fundamental frequency: {fundamental_frequency:.1f} Hz")
    
    # Generate synthetic data for each bearing
    def generate_bearing_data(angles, rolling_elements, bearing_name):
        print(f"   Generating {bearing_name} data...")
        
        num_sensors = len(angles)
        fft_mags = np.zeros((len(freqs), num_sensors))
        fft_phases = np.zeros((len(freqs), num_sensors))
        
        for sensor_idx, angle in enumerate(angles):
            # Create synthetic signal with fundamental + harmonics + noise
            signal = np.zeros(num_samples)
            
            # Fundamental frequency component
            signal += 0.5 * np.sin(2 * np.pi * fundamental_frequency * time_vector + np.radians(angle))
            
            # Add harmonics with decreasing amplitude
            for harmonic in [2, 3, 4, 5]:
                amplitude = 0.3 / harmonic
                phase_shift = np.radians(angle * harmonic / 2)
                signal += amplitude * np.sin(2 * np.pi * fundamental_frequency * harmonic * time_vector + phase_shift)
            
            # Add rolling element frequencies
            rolling_freq = fundamental_frequency * rolling_elements / 2
            signal += 0.2 * np.sin(2 * np.pi * rolling_freq * time_vector + np.radians(angle))
            
            # Add noise
            signal += 0.1 * np.random.randn(num_samples)
            
            # Compute FFT
            fft_result = np.fft.fft(signal)
            fft_result = fft_result[:len(freqs)]  # Only positive frequencies
            
            # Store magnitude and phase
            fft_mags[:, sensor_idx] = np.abs(fft_result)
            fft_phases[:, sensor_idx] = np.angle(fft_result, deg=True)
        
        return {
            'fft_mags': fft_mags,
            'fft_phases': fft_phases,
            'angles': angles
        }
    
    # Generate data for both bearings
    bearing_a_data = generate_bearing_data(bearing_a_angles, bearing_a_rolling_elements, "Bearing A")
    bearing_b_data = generate_bearing_data(bearing_b_angles, bearing_b_rolling_elements, "Bearing B")
    
    # Save to HDF5 file
    output_file = 'sample_dual_bearing_data.h5'
    print(f"💾 Saving sample data to {output_file}...")
    
    with h5py.File(output_file, 'w') as h5f:
        # Time and frequency data
        h5f.create_dataset('time_vector', data=time_vector)
        h5f.create_dataset('freqs', data=freqs)
        
        # Bearing A data
        bearing_a_group = h5f.create_group('bearing_a')
        bearing_a_group.create_dataset('fft_mags', data=bearing_a_data['fft_mags'])
        bearing_a_group.create_dataset('fft_phases', data=bearing_a_data['fft_phases'])
        bearing_a_group.create_dataset('sensor_angles', data=bearing_a_data['angles'])
        bearing_a_group.attrs['rolling_elements'] = bearing_a_rolling_elements
        
        # Bearing B data
        bearing_b_group = h5f.create_group('bearing_b')
        bearing_b_group.create_dataset('fft_mags', data=bearing_b_data['fft_mags'])
        bearing_b_group.create_dataset('fft_phases', data=bearing_b_data['fft_phases'])
        bearing_b_group.create_dataset('sensor_angles', data=bearing_b_data['angles'])
        bearing_b_group.attrs['rolling_elements'] = bearing_b_rolling_elements
        
        # Metadata
        h5f.attrs['sampling_frequency'] = sampling_frequency
        h5f.attrs['shaft_speed_rpm'] = shaft_speed_rpm
        h5f.attrs['fundamental_frequency_hz'] = fundamental_frequency
    
    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Sample data generated successfully!")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   This lightweight dataset is suitable for Streamlit Cloud deployment.")
    
    return output_file

if __name__ == "__main__":
    generate_sample_data() 