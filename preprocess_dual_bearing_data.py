import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import h5py
import os
from datetime import datetime

def load_mat_file(file_path):
    """Load the .mat file containing fiber sensor data"""
    print(f"Loading .mat file: {file_path}")
    return loadmat(file_path)

def create_sensor_info():
    """Create sensor information for both bearings"""
    # Bearing A sensor information (columns 1-13 in wl_fos, but we'll use 0-based indexing)
    bearing_a_sensors = list(range(12, -1, -1))  # 12 to 0 (0-based indexing for columns 13 to 1)
    bearing_a_angles = [67.000, 94.692, 122.385, 150.077, 177.769, 205.462, 233.154,
                       260.846, 288.538, 316.231, 343.923, 371.615, 399.308]
    
    # Bearing B sensor information (columns 14-26 in wl_fos, so 13-25 in 0-based indexing)
    bearing_b_sensors = list(range(25, 12, -1))  # 25 to 13 (0-based indexing for columns 26 to 14)
    bearing_b_angles = [282.0000, 254.3077, 226.6154, 198.9231, 171.2308, 143.5385,
                       115.8462, 88.1538, 60.4615, 32.7692, 5.0769, -22.6154, -50.3077]
    
    # Print sensor mapping information
    print("\nBearing A Sensor Mapping:")
    print("Sensor Number (1-based) | Angle (degrees)")
    print("-" * 35)
    for idx, (sensor, angle) in enumerate(zip(bearing_a_sensors, bearing_a_angles)):
        print(f"Sensor {13-idx:2d}           | {angle:8.3f}")
        
    print("\nBearing B Sensor Mapping:")  
    print("Sensor Number (1-based) | Angle (degrees)")
    print("-" * 35)
    for idx, (sensor, angle) in enumerate(zip(bearing_b_sensors, bearing_b_angles)):
        print(f"Sensor {26-idx:2d}           | {angle:8.3f}")
        
    print("\nRolling Elements:")
    print("- Bearing A has 19 rolling elements for detecting periodic impacts")
    print("- Bearing B has 21 rolling elements for detecting periodic impacts")
    
    # Create sensor info dictionary
    sensor_info = {
        'bearing_a': {
            'sensor_indices': bearing_a_sensors,
            'angles': bearing_a_angles,
            'rolling_elements': 19  # Used for calculating expected fault frequencies
        },
        'bearing_b': {
            'sensor_indices': bearing_b_sensors,
            'angles': bearing_b_angles,
            'rolling_elements': 21  # Used for calculating expected fault frequencies
        }
    }
    
    return sensor_info

def create_equidistant_time_series(time_data, target_fs=2000):
    """Create equidistant time series and interpolate data"""
    print("Creating equidistant time series...")
    
    # Get time range
    start_time = time_data[0]
    end_time = time_data[-1]
    total_duration = end_time - start_time
    
    # Make it a whole number of seconds for cleaner processing
    total_duration_whole = int(total_duration)
    actual_end_time = start_time + total_duration_whole
    
    # Create equidistant time vector
    num_samples = int(total_duration_whole * target_fs)
    equidistant_time = np.linspace(start_time, actual_end_time, num_samples)
    
    print(f"Original time range: {start_time:.3f} to {end_time:.3f} seconds")
    print(f"New time range: {start_time:.3f} to {actual_end_time:.3f} seconds")
    print(f"Duration: {total_duration_whole:.1f} seconds")
    print(f"Number of samples: {num_samples}")
    print(f"Sampling frequency: {target_fs} Hz")
    
    return equidistant_time

def interpolate_sensor_data(original_time, new_time, sensor_data):
    """Interpolate sensor data to new time series"""
    print("Interpolating sensor data...")
    
    num_sensors = sensor_data.shape[1]
    interpolated_data = np.zeros((len(new_time), num_sensors))
    
    for sensor_idx in range(num_sensors):
        # Create interpolation function
        interp_func = interp1d(original_time, sensor_data[:, sensor_idx], 
                              kind='linear', bounds_error=False)
        
        # Interpolate to new time series
        interpolated_data[:, sensor_idx] = interp_func(new_time)
        
        if sensor_idx % 5 == 0:  # Progress indicator
            print(f"Interpolated sensor {sensor_idx + 1}/{num_sensors}")
    
    return interpolated_data

def perform_fft_analysis(data, fs=2000):
    """
    Compute the FFT of the given data after removing mean and applying window.
    
    Parameters:
        data (numpy.ndarray): A 2D array where each row is a time sample, each column is a sensor.
        fs (float): Sampling frequency in Hz
            
    Returns:
        tuple: (frequencies, FFT magnitude array, FFT phase array)
    """
    print("Performing FFT analysis...")
    
    n_samples, n_sensors = data.shape
    
    # Create frequency bins
    freqs = fftfreq(n_samples, d=1/fs)[:n_samples//2]  # Only positive frequencies
    
    # Initialize arrays to hold the FFT mags and phases
    fft_mags = np.zeros((len(freqs), n_sensors))
    fft_phases = np.zeros((len(freqs), n_sensors))
    
    # Apply window
    window = np.hanning(n_samples)
    window_norm = np.sum(window)
    
    # Compute FFT for each sensor
    for sensor_idx in range(n_sensors):
        signal = data[:, sensor_idx]
        
        # Remove mean
        signal = signal - np.mean(signal)
        
        # Apply window
        windowed_signal = signal * window
        
        # Compute FFT
        fft_result = fft(windowed_signal)
        
        # Take only positive frequencies
        fft_result = fft_result[:len(freqs)]
        
        # Calculate magnitude and phase
        mags = (2.0 / window_norm) * np.abs(fft_result)
        phases = np.angle(fft_result, deg=True)
        
        fft_mags[:, sensor_idx] = mags
        fft_phases[:, sensor_idx] = phases
        
        if sensor_idx % 5 == 0:  # Progress indicator
            print(f"FFT completed for sensor {sensor_idx + 1}/{n_sensors}")
    
    return freqs, fft_mags, fft_phases

def save_processed_data(output_file, time_vector, freqs, fft_mags, fft_phases, 
                       sensor_info, interpolated_data=None):
    """Save all processed data to HDF5 file"""
    print(f"Saving processed data to: {output_file}")
    
    with h5py.File(output_file, 'w') as h5f:
        # Save time and frequency vectors
        h5f.create_dataset('time_vector', data=time_vector)
        h5f.create_dataset('freqs', data=freqs)
        
        # Save FFT data for bearing A
        bearing_a_group = h5f.create_group('bearing_a')
        bearing_a_indices = sensor_info['bearing_a']['sensor_indices']
        
        bearing_a_group.create_dataset('fft_mags', data=fft_mags[:, bearing_a_indices])
        bearing_a_group.create_dataset('fft_phases', data=fft_phases[:, bearing_a_indices])
        bearing_a_group.create_dataset('sensor_angles', data=sensor_info['bearing_a']['angles'])
        bearing_a_group.attrs['rolling_elements'] = sensor_info['bearing_a']['rolling_elements']
        bearing_a_group.attrs['num_sensors'] = len(bearing_a_indices)
        
        # Save FFT data for bearing B
        bearing_b_group = h5f.create_group('bearing_b')
        bearing_b_indices = sensor_info['bearing_b']['sensor_indices']
        
        bearing_b_group.create_dataset('fft_mags', data=fft_mags[:, bearing_b_indices])
        bearing_b_group.create_dataset('fft_phases', data=fft_phases[:, bearing_b_indices])
        bearing_b_group.create_dataset('sensor_angles', data=sensor_info['bearing_b']['angles'])
        bearing_b_group.attrs['rolling_elements'] = sensor_info['bearing_b']['rolling_elements']
        bearing_b_group.attrs['num_sensors'] = len(bearing_b_indices)
        
        # Optionally save interpolated raw data (if needed for debugging)
        if interpolated_data is not None:
            raw_group = h5f.create_group('raw_data')
            raw_group.create_dataset('bearing_a_data', data=interpolated_data[:, bearing_a_indices])
            raw_group.create_dataset('bearing_b_data', data=interpolated_data[:, bearing_b_indices])
        
        # Add metadata
        h5f.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        h5f.attrs['description'] = 'Processed dual bearing fiber sensor data'
        h5f.attrs['sampling_frequency'] = 2000.0
        h5f.attrs['shaft_speed_rpm'] = 450.0
        h5f.attrs['fundamental_frequency_hz'] = 450.0 / 60.0  # 7.5 Hz
        
        # Add units
        h5f['time_vector'].attrs['units'] = 'seconds'
        h5f['freqs'].attrs['units'] = 'Hz'
        bearing_a_group['fft_mags'].attrs['units'] = 'magnitude'
        bearing_a_group['fft_phases'].attrs['units'] = 'degrees'
        bearing_b_group['fft_mags'].attrs['units'] = 'magnitude'
        bearing_b_group['fft_phases'].attrs['units'] = 'degrees'
        bearing_a_group['sensor_angles'].attrs['units'] = 'degrees'
        bearing_b_group['sensor_angles'].attrs['units'] = 'degrees'
    
    print("Data successfully saved!")

def plot_validation_figures(freqs, fft_mags, sensor_info, output_dir):
    """Create validation plots to check the processing"""
    print("Creating validation plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot FFT for first few sensors of each bearing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bearing A
    bearing_a_indices = sensor_info['bearing_a']['sensor_indices'][:3]  # First 3 sensors
    for i, sensor_idx in enumerate(bearing_a_indices):
        ax1.plot(freqs, fft_mags[:, sensor_idx], label=f'Sensor {sensor_idx+1}')
    ax1.set_title('Bearing A - FFT Magnitude (First 3 Sensors)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xlim(0, 100)  # Focus on low frequencies
    
    # Bearing B
    bearing_b_indices = sensor_info['bearing_b']['sensor_indices'][:3]  # First 3 sensors
    for i, sensor_idx in enumerate(bearing_b_indices):
        ax2.plot(freqs, fft_mags[:, sensor_idx], label=f'Sensor {sensor_idx+1}')
    ax2.set_title('Bearing B - FFT Magnitude (First 3 Sensors)')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.legend()
    ax2.grid(True)
    ax2.set_xlim(0, 100)  # Focus on low frequencies
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fft_validation_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a simple bearing layout plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bearing A layout
    angles_a = np.array(sensor_info['bearing_a']['angles'])
    angles_a_rad = np.radians(angles_a)
    ax1.scatter(np.cos(angles_a_rad), np.sin(angles_a_rad), s=100, c='red', alpha=0.7)
    for i, angle in enumerate(angles_a):
        ax1.annotate(f'S{i+1}', (np.cos(angles_a_rad[i]), np.sin(angles_a_rad[i])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Bearing A - Sensor Layout')
    ax1.grid(True, alpha=0.3)
    
    # Bearing B layout
    angles_b = np.array(sensor_info['bearing_b']['angles'])
    angles_b_rad = np.radians(angles_b)
    ax2.scatter(np.cos(angles_b_rad), np.sin(angles_b_rad), s=100, c='blue', alpha=0.7)
    for i, angle in enumerate(angles_b):
        ax2.annotate(f'S{i+1}', (np.cos(angles_b_rad[i]), np.sin(angles_b_rad[i])), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Bearing B - Sensor Layout')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sensor_layout.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Validation plots saved!")

def main():
    """Main preprocessing function"""
    print("=== Dual Bearing Fiber Sensor Data Preprocessing ===")
    
    # File paths
    input_file = 'optics11_recording_2024_02_08_09_27_48.mat'
    output_file = 'processed_dual_bearing_data.h5'
    output_dir = 'validation_plots'
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
    
    # Load the .mat file
    mat_data = load_mat_file(input_file)
    
    # Extract data
    wl_fos = mat_data['wl_fos']  # Wavelength data
    time_fos = mat_data['time_fos'].flatten()  # Time stamps
    
    print(f"Original data shape: {wl_fos.shape}")
    print(f"Time vector length: {len(time_fos)}")
    print(f"Time range: {time_fos[0]:.3f} to {time_fos[-1]:.3f} seconds")
    
    # Create sensor information
    sensor_info = create_sensor_info()
    
    # Create equidistant time series
    new_time_vector = create_equidistant_time_series(time_fos, target_fs=2000)
    
    # Interpolate data to new time series
    interpolated_data = interpolate_sensor_data(time_fos, new_time_vector, wl_fos)
    
    print(f"Interpolated data shape: {interpolated_data.shape}")
    
    # Perform FFT analysis
    freqs, fft_mags, fft_phases = perform_fft_analysis(interpolated_data, fs=2000)
    
    print(f"FFT data shape: {fft_mags.shape}")
    print(f"Frequency range: {freqs[0]:.3f} to {freqs[-1]:.3f} Hz")
    print(f"Fundamental frequency (450 RPM): {450/60:.1f} Hz")
    
    # Save processed data
    save_processed_data(output_file, new_time_vector, freqs, fft_mags, fft_phases, 
                       sensor_info, interpolated_data)
    
    # Create validation plots
    plot_validation_figures(freqs, fft_mags, sensor_info, output_dir)
    
    print("\n=== Processing Complete! ===")
    print(f"Processed data saved to: {output_file}")
    print(f"Validation plots saved to: {output_dir}/")
    
    # Print summary
    print("\nData Summary:")
    print(f"- Total sensors: {wl_fos.shape[1]} (13 per bearing)")
    print(f"- Bearing A: 13 sensors, 19 rolling elements")
    print(f"- Bearing B: 13 sensors, 21 rolling elements")
    print(f"- Sampling frequency: 2000 Hz")
    print(f"- Fundamental frequency: {450/60:.1f} Hz")
    print(f"- Processed time duration: {len(new_time_vector)/2000:.1f} seconds")

if __name__ == "__main__":
    main() 