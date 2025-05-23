import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import h5py
import plotly.io as pio
from scipy.interpolate import splprep, splev
import gc
import datetime

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Dual Bearing FFT Visualization")

# --- GENERATE SAMPLE DATA FUNCTION ---
def generate_sample_data_for_demo():
    """Generate lightweight sample data for demo purposes"""
    # Parameters
    sampling_frequency = 2000  # Hz
    duration = 5  # seconds (very small for demo)
    shaft_speed_rpm = 450
    fundamental_frequency = shaft_speed_rpm / 60  # 7.5 Hz
    
    # Time vector
    num_samples = int(sampling_frequency * duration)
    time_vector = np.linspace(0, duration, num_samples)
    
    # Frequency vector for FFT
    freqs = np.fft.fftfreq(num_samples, 1/sampling_frequency)
    freqs = freqs[:num_samples//2]  # Only positive frequencies
    
    # Bearing configurations
    bearing_a_angles = np.array([67.0, 94.7, 122.4, 150.1, 177.8, 205.5, 233.2, 260.8, 288.5, 316.2, 343.9, 371.6, 399.3])
    bearing_b_angles = np.array([282.0, 254.3, 226.6, 198.9, 171.2, 143.5, 115.8, 88.2, 60.5, 32.8, 5.1, -22.6, -50.3])
    
    # Generate synthetic FFT data
    def generate_bearing_fft(angles, rolling_elements):
        num_sensors = len(angles)
        fft_mags = np.zeros((len(freqs), num_sensors))
        fft_phases = np.zeros((len(freqs), num_sensors))
        
        for sensor_idx, angle in enumerate(angles):
            # Create synthetic FFT peaks at key frequencies
            for freq_mult in [1, 2, 3, 4, 5, 8]:
                target_freq = fundamental_frequency * freq_mult
                freq_idx = np.argmin(np.abs(freqs - target_freq))
                
                # Add magnitude and phase based on sensor position
                amplitude = 1.0 / freq_mult * (0.5 + 0.5 * np.random.random())
                phase = angle + np.random.normal(0, 10)  # Some phase variation
                
                # Add some width to the peak
                for offset in range(-2, 3):
                    if 0 <= freq_idx + offset < len(freqs):
                        fft_mags[freq_idx + offset, sensor_idx] += amplitude * np.exp(-offset**2/2)
                        fft_phases[freq_idx + offset, sensor_idx] = phase
        
        return fft_mags, fft_phases
    
    # Generate data for both bearings
    bearing_a_fft_mags, bearing_a_fft_phases = generate_bearing_fft(bearing_a_angles, 19)
    bearing_b_fft_mags, bearing_b_fft_phases = generate_bearing_fft(bearing_b_angles, 21)
    
    return {
        'time_vector': time_vector,
        'freqs': freqs,
        'bearing_a': {
            'fft_mags': bearing_a_fft_mags,
            'fft_phases': bearing_a_fft_phases,
            'angles': bearing_a_angles,
            'rolling_elements': 19
        },
        'bearing_b': {
            'fft_mags': bearing_b_fft_mags,
            'fft_phases': bearing_b_fft_phases,
            'angles': bearing_b_angles,
            'rolling_elements': 21
        },
        'sampling_frequency': sampling_frequency,
        'shaft_speed_rpm': shaft_speed_rpm,
        'fundamental_frequency': fundamental_frequency
    }

# --- LOAD PROCESSED DATA ---
@st.cache_data(ttl=24*3600, max_entries=1)
def load_processed_data():
    """Load the processed dual bearing data from HDF5 file or generate sample data"""
    
    # Try different data file options
    data_files = [
        'processed_dual_bearing_data.h5',
        'sample_dual_bearing_data.h5'
    ]
    
    data_file = None
    for file in data_files:
        if os.path.exists(file):
            data_file = file
            break
    
    if data_file is None:
        # No data file found, use sample data for demo
        st.info("📊 **Demo Mode**: Using synthetic sample data for demonstration. Upload your own data file for full functionality.")
        return generate_sample_data_for_demo()
    
    # Load data from file
    with st.spinner(f"Loading data from {data_file}..."):
        try:
            with h5py.File(data_file, 'r') as h5f:
                # Load time and frequency data
                time_vector = h5f['time_vector'][:]
                freqs = h5f['freqs'][:]
                
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
                
                # Load metadata
                sampling_frequency = h5f.attrs['sampling_frequency']
                shaft_speed_rpm = h5f.attrs['shaft_speed_rpm']
                fundamental_frequency = h5f.attrs['fundamental_frequency_hz']
                
            st.success(f"✅ Loaded data from {data_file}")
            
        except Exception as e:
            st.error(f"Error loading data file {data_file}: {e}")
            st.info("Using sample data instead...")
            return generate_sample_data_for_demo()
    
    return {
        'time_vector': time_vector,
        'freqs': freqs,
        'bearing_a': {
            'fft_mags': bearing_a_fft_mags,
            'fft_phases': bearing_a_fft_phases,
            'angles': bearing_a_angles,
            'rolling_elements': bearing_a_rolling_elements
        },
        'bearing_b': {
            'fft_mags': bearing_b_fft_mags,
            'fft_phases': bearing_b_fft_phases,
            'angles': bearing_b_angles,
            'rolling_elements': bearing_b_rolling_elements
        },
        'sampling_frequency': sampling_frequency,
        'shaft_speed_rpm': shaft_speed_rpm,
        'fundamental_frequency': fundamental_frequency
    }

# Load data
data = load_processed_data()

# Extract data
time_vector = data['time_vector']
freqs = data['freqs']
fundamental_freq = data['fundamental_frequency']

# Process time (convert if needed)
if time_vector[0] > 1e17:
    time_vector = time_vector / 1e9

time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- PAGE HEADER ---
st.title("🔧 Dual Bearing Fiber Sensor FFT Visualization")
st.markdown(f"**Shaft Speed:** {data['shaft_speed_rpm']} RPM | **Fundamental Frequency:** {fundamental_freq:.1f} Hz | **Sampling Rate:** {data['sampling_frequency']} Hz")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Visualization Controls")

# File upload option
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload HDF5 data file",
    type=['h5'],
    help="Upload a processed dual bearing data file in HDF5 format"
)

if uploaded_file is not None:
    st.sidebar.success("File uploaded! Please refresh to use the new data.")

# Frequency selection - using bearing-specific frequencies
bearing_freqs_to_plot = [
    fundamental_freq,  # 7.5 Hz (450 RPM fundamental)
    fundamental_freq * 2,  # 15.0 Hz (2x fundamental)
    fundamental_freq * 3,  # 22.5 Hz (3x fundamental)
    fundamental_freq * 4,  # 30.0 Hz (4x fundamental)
    fundamental_freq * 5,  # 37.5 Hz (5x fundamental)
    fundamental_freq * 8,  # 60.0 Hz (8x fundamental)
    fundamental_freq * 12, # 90.0 Hz (12x fundamental)
    fundamental_freq * 16  # 120.0 Hz (16x fundamental)
]

freq_labels = [f"{freq:.1f} Hz ({int(freq/fundamental_freq)}x)" for freq in bearing_freqs_to_plot]
freq_selected_idx = st.sidebar.selectbox("Select Frequency", range(len(freq_labels)), 
                                        format_func=lambda x: freq_labels[x], index=1)
freq_selected = bearing_freqs_to_plot[freq_selected_idx]

# Animation settings
animation_fps = st.sidebar.slider("Animation FPS", min_value=2, max_value=10, value=5, step=1)

# Time index slider
time_idx = st.sidebar.slider("Time Index", min_value=0, max_value=num_times-1, value=0)

# Display selected time
st.sidebar.info(f"**Selected Time:** {time_formatted[time_idx].strftime('%Y-%m-%d %H:%M:%S')}")

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=300, max_entries=50)
def find_peak_magnitude_bearing(bearing_data, freq_target, time_idx, window=0.2):
    """Find peak magnitude and phase for all sensors in a bearing at a specific frequency and time"""
    fft_mags = bearing_data['fft_mags']
    fft_phases = bearing_data['fft_phases']
    
    # Find frequency indices within the window
    freq_mask = (freqs >= freq_target - window) & (freqs <= freq_target + window)
    freq_indices = np.where(freq_mask)[0]
    
    if len(freq_indices) == 0:
        return [0] * fft_mags.shape[1], [0] * fft_phases.shape[1]
    
    magnitudes = []
    phases = []
    
    for sensor_idx in range(fft_mags.shape[1]):
        # Get magnitude values in the frequency window
        mag_vals = fft_mags[freq_indices, sensor_idx]
        max_idx = np.argmax(mag_vals)
        
        # Get corresponding phase
        phase_vals = fft_phases[freq_indices, sensor_idx]
        
        magnitudes.append(mag_vals[max_idx])
        phases.append(phase_vals[max_idx])
    
    return magnitudes, phases

def create_animated_bearing_plot(bearing_data, bearing_name, freq_selected, time_idx, fps=5):
    """Create animated bearing plot for a single bearing"""
    
    # Get magnitude and phase values
    magnitudes, phases = find_peak_magnitude_bearing(bearing_data, freq_selected, time_idx)
    sensor_angles = bearing_data['angles']
    rolling_elements = bearing_data['rolling_elements']
    
    # Convert angles to radians
    sensor_angles_rad = np.radians(sensor_angles)
    
    # Define sensor IDs based on bearing
    if "A" in bearing_name:
        # Bearing A: sensor indices 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
        sensor_ids = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    else:
        # Bearing B: sensor indices 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14
        sensor_ids = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14]
    
    # Create base figure
    fig = go.Figure()
    
    # Calculate period for animation
    T = 1.0 / freq_selected
    num_points = 100
    tvec = np.linspace(0, 3*T, num_points)  # 3 cycles for better visualization
    
    # Create animation frames
    frames = []
    for frame_idx, t in enumerate(tvec):
        # Calculate current radial values based on sine wave
        radii = []
        max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0
        
        for mag, phase in zip(magnitudes, phases):
            # x(t) = A sin(wt + phi)
            val = mag * np.sin(2 * np.pi * freq_selected * t + np.radians(phase))
            # Normalize and offset for positive display
            scaled_val = 1 + (val / max_mag) * 0.8  # Scale to 0.2 to 1.8 range
            radii.append(scaled_val)
        
        # Create frame data
        frame_data = []
        
        # Add bearing circle (outer ring)
        circle_theta = np.linspace(0, 360, 361)
        frame_data.append(
            go.Scatterpolar(
                r=[1]*361,
                theta=circle_theta,
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                showlegend=False,
                name='Bearing Ring'
            )
        )
        
        # Add sensor positions
        sensor_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
                        'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
        
        for i, (angle, radius, sensor_id) in enumerate(zip(sensor_angles, radii, sensor_ids)):
            color = sensor_colors[i % len(sensor_colors)]
            
            # Sensor dot
            frame_data.append(
                go.Scatterpolar(
                    r=[radius],
                    theta=[angle],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    showlegend=False,
                    name=f'Sensor {sensor_id}'
                )
            )
            
            # Magnitude line from center
            frame_data.append(
                go.Scatterpolar(
                    r=[0, radius],
                    theta=[angle, angle],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False
                )
            )
            
            # Add sensor ID label at the end of the line
            label_radius = radius + 0.15  # Offset the label slightly beyond the sensor
            frame_data.append(
                go.Scatterpolar(
                    r=[label_radius],
                    theta=[angle],
                    mode='text',
                    text=[str(sensor_id)],
                    textfont=dict(size=10, color=color),
                    showlegend=False,
                    name=f'Label {sensor_id}'
                )
            )
        
        # Add connecting lines between sensors
        ordered_radii = radii + [radii[0]]  # Close the loop
        ordered_angles = list(sensor_angles) + [sensor_angles[0]]
        
        frame_data.append(
            go.Scatterpolar(
                r=ordered_radii,
                theta=ordered_angles,
                mode='lines',
                line=dict(color='blue', width=2, dash='dot'),
                showlegend=False,
                name='Sensor Connection'
            )
        )
        
        # Try to add smooth interpolated curve
        if len(sensor_angles) >= 4:  # Need at least 4 points for interpolation
            try:
                # Convert to cartesian for interpolation
                x_points = np.array([r * np.cos(np.radians(theta)) for r, theta in zip(radii, sensor_angles)])
                y_points = np.array([r * np.sin(np.radians(theta)) for r, theta in zip(radii, sensor_angles)])
                
                # Close the curve
                x_closed = np.append(x_points, x_points[0])
                y_closed = np.append(y_points, y_points[0])
                
                # Interpolate
                tck, u = splprep([x_closed, y_closed], s=0, per=True, k=3)
                u_new = np.linspace(0, 1, 200)
                x_smooth, y_smooth = splev(u_new, tck)
                
                # Convert back to polar
                r_smooth = np.sqrt(x_smooth**2 + y_smooth**2)
                theta_smooth = np.degrees(np.arctan2(y_smooth, x_smooth)) % 360
                
                frame_data.append(
                    go.Scatterpolar(
                        r=r_smooth,
                        theta=theta_smooth,
                        mode='lines',
                        line=dict(color='rgba(255,0,0,0.7)', width=2),
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False,
                        name='Interpolated Shape'
                    )
                )
            except Exception:
                pass  # Skip interpolation if it fails
        
        # Create frame
        frames.append(go.Frame(data=frame_data, name=f"frame{frame_idx}"))
    
    # Add initial data (first frame)
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)
    
    # Configure animation
    fig.frames = frames
    
    # Calculate frame duration
    frame_duration = max(50, (T * 1000) / len(frames))  # At least 50ms per frame
    
    # Animation settings
    animation_settings = dict(
        frame=dict(duration=frame_duration, redraw=True),
        fromcurrent=True,
        mode="immediate"
    )
    
    # Layout configuration
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶️ Play",
                        method="animate",
                        args=[None, animation_settings]
                    ),
                    dict(
                        label="⏸️ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=True), mode="immediate")]
                    )
                ],
                direction="left",
                pad=dict(r=10, t=10),
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2.5],  # Increased range to accommodate labels
                showticklabels=False
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                direction="clockwise",
                rotation=90
            )
        ),
        title=dict(
            text=f"{bearing_name}<br>{freq_selected:.1f} Hz (T={T:.3f}s) | {rolling_elements} rolling elements",
            font=dict(size=14),
            x=0.5
        ),
        height=500,
        width=500,
        margin=dict(t=80, l=20, r=20, b=20)
    )
    
    return fig

@st.cache_data(ttl=60, max_entries=10)
def create_fft_spectrum_plot(bearing_data, bearing_name, time_idx, freq_selected):
    """Create FFT spectrum plot for a bearing"""
    
    fft_mags = bearing_data['fft_mags']
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for sensors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']
    
    # Plot each sensor
    for sensor_idx in range(fft_mags.shape[1]):
        fig.add_trace(go.Scatter(
            x=freqs,
            y=fft_mags[:, sensor_idx],
            name=f'S{sensor_idx+1}',
            line=dict(color=colors[sensor_idx % len(colors)], width=1.5),
            opacity=0.7
        ))
    
    # Add vertical line at selected frequency
    fig.add_vline(
        x=freq_selected, 
        line=dict(color="red", width=2, dash="dash"),
        annotation_text=f"{freq_selected:.1f} Hz"
    )
    
    # Add fundamental frequency markers
    for i in range(1, 6):
        freq_marker = fundamental_freq * i
        if freq_marker <= freqs[-1]:
            fig.add_vline(
                x=freq_marker,
                line=dict(color="green", width=1, dash="dot"),
                opacity=0.5
            )
    
    # Layout
    fig.update_layout(
        title=f"{bearing_name} - FFT Spectrum",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(range=[0, 200])  # Focus on lower frequencies
    )
    
    return fig

# --- MAIN DASHBOARD LAYOUT ---

# Display current settings
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Selected Frequency", f"{freq_selected:.1f} Hz")
with col_info2:
    st.metric("Time Index", f"{time_idx}/{num_times-1}")
with col_info3:
    st.metric("Animation FPS", animation_fps)

st.markdown("---")

# Bearing visualization section
st.subheader("🔄 Animated Bearing Visualization")

# Create two columns for side-by-side bearing plots
col_bearing_a, col_bearing_b = st.columns(2)

with col_bearing_a:
    st.markdown("### Bearing A (19 rolling elements)")
    bearing_a_fig = create_animated_bearing_plot(
        data['bearing_a'], "Bearing A", freq_selected, time_idx, animation_fps
    )
    st.plotly_chart(bearing_a_fig, use_container_width=True)

with col_bearing_b:
    st.markdown("### Bearing B (21 rolling elements)")
    bearing_b_fig = create_animated_bearing_plot(
        data['bearing_b'], "Bearing B", freq_selected, time_idx, animation_fps
    )
    st.plotly_chart(bearing_b_fig, use_container_width=True)

st.markdown("---")

# FFT spectrum section
st.subheader("📊 FFT Spectrum Analysis")

# Create two columns for side-by-side FFT plots
col_fft_a, col_fft_b = st.columns(2)

with col_fft_a:
    fft_a_fig = create_fft_spectrum_plot(data['bearing_a'], "Bearing A", time_idx, freq_selected)
    st.plotly_chart(fft_a_fig, use_container_width=True)

with col_fft_b:
    fft_b_fig = create_fft_spectrum_plot(data['bearing_b'], "Bearing B", time_idx, freq_selected)
    st.plotly_chart(fft_b_fig, use_container_width=True)

# --- ADDITIONAL INFO ---
st.markdown("---")

# Expandable technical information
with st.expander("🔍 Technical Information"):
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("**Bearing A:**")
        st.write(f"- Sensors: {len(data['bearing_a']['angles'])}")
        st.write(f"- Rolling elements: {data['bearing_a']['rolling_elements']}")
        st.write(f"- Sensor angles: {data['bearing_a']['angles']}")
    
    with col_tech2:
        st.markdown("**Bearing B:**")
        st.write(f"- Sensors: {len(data['bearing_b']['angles'])}")
        st.write(f"- Rolling elements: {data['bearing_b']['rolling_elements']}")
        st.write(f"- Sensor angles: {data['bearing_b']['angles']}")

# Data information
with st.expander("📈 Data Information"):
    st.write(f"**Total time duration:** {len(time_vector)/data['sampling_frequency']:.1f} seconds")
    st.write(f"**Frequency resolution:** {freqs[1] - freqs[0]:.3f} Hz")
    st.write(f"**Maximum frequency:** {freqs[-1]:.1f} Hz")
    st.write(f"**Time vector shape:** {time_vector.shape}")
    st.write(f"**FFT data shape (Bearing A):** {data['bearing_a']['fft_mags'].shape}")
    st.write(f"**FFT data shape (Bearing B):** {data['bearing_b']['fft_mags'].shape}")

# Footer
st.markdown("---")
st.markdown("*Dashboard created for dual bearing fiber sensor FFT analysis*") 