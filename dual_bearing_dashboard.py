import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import h5py
import plotly.io as pio
from scipy.interpolate import splprep, splev
import gc

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Dual Bearing FFT Visualization")

# --- LOAD PROCESSED DATA ---
@st.cache_data(ttl=24*3600, max_entries=1)
def load_processed_data():
    """Load the processed dual bearing data from HDF5 file"""
    data_file = 'processed_dual_bearing_data.h5'
    
    if not os.path.exists(data_file):
        st.error(f"Processed data file '{data_file}' not found! Please run the preprocessing script first.")
        st.info("For Streamlit Cloud deployment, ensure the data file is available or use the file upload feature.")
        st.stop()
    
    with st.spinner("Loading processed dual bearing data..."):
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
            st.stop()
    
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
animation_fps = st.sidebar.slider("Animation FPS", min_value=2, max_value=8, value=4, step=1)

# Time index slider
time_idx = st.sidebar.slider("Time Index", min_value=0, max_value=num_times-1, value=0)

# Display selected time
st.sidebar.info(f"**Selected Time:** {time_formatted[time_idx].strftime('%Y-%m-%d %H:%M:%S')}")

# --- MEMORY-OPTIMIZED HELPER FUNCTIONS ---
@st.cache_data(ttl=300, max_entries=20)
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

def create_memory_optimized_bearing_plot(bearing_data, bearing_name, freq_selected, time_idx, fps=4):
    """Create memory-optimized bearing plot with reduced animation frames"""
    
    # Get magnitude and phase values
    magnitudes, phases = find_peak_magnitude_bearing(bearing_data, freq_selected, time_idx)
    sensor_angles = bearing_data['angles']
    rolling_elements = bearing_data['rolling_elements']
    
    # Define sensor IDs based on bearing
    if "A" in bearing_name:
        sensor_ids = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    else:
        sensor_ids = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14]
    
    # Create base figure
    fig = go.Figure()
    
    # Calculate period for animation - REDUCED FRAMES FOR MEMORY
    T = 1.0 / freq_selected
    num_points = 30  # Reduced from 100 to 30 for memory efficiency
    tvec = np.linspace(0, 2*T, num_points)  # Reduced from 3T to 2T
    
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
        
        # Create frame data - SIMPLIFIED FOR MEMORY
        frame_data = []
        
        # Add bearing circle (outer ring) - reduced points
        circle_theta = np.linspace(0, 360, 73)  # Reduced from 361 to 73
        frame_data.append(
            go.Scatterpolar(
                r=[1]*73,
                theta=circle_theta,
                mode='lines',
                line=dict(color='gray', dash='dot', width=1),
                showlegend=False,
                name='Bearing Ring'
            )
        )
        
        # Add sensor positions - SIMPLIFIED
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
                    marker=dict(size=8, color=color),
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
                    line=dict(color=color, width=2),
                    showlegend=False
                )
            )
        
        # Add simple connecting lines (no interpolation for memory efficiency)
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
        
        # Create frame
        frames.append(go.Frame(data=frame_data, name=f"frame{frame_idx}"))
    
    # Add initial data (first frame)
    if frames:
        for trace in frames[0].data:
            fig.add_trace(trace)
    
    # Configure animation
    fig.frames = frames
    
    # Calculate frame duration
    frame_duration = max(100, (T * 1000) / len(frames))  # Minimum 100ms per frame
    
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
                range=[0, 2.2],
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
        height=450,  # Reduced height
        width=450,   # Reduced width
        margin=dict(t=80, l=20, r=20, b=20)
    )
    
    return fig

@st.cache_data(ttl=60, max_entries=10)
def create_memory_optimized_fft_plot(bearing_data, bearing_name, time_idx, freq_selected):
    """Create memory-optimized FFT spectrum plot"""
    
    fft_mags = bearing_data['fft_mags']
    
    # MEMORY OPTIMIZATION: Plot every 10th frequency point instead of all
    freq_step = 10
    freq_indices = np.arange(0, len(freqs), freq_step)
    freqs_reduced = freqs[freq_indices]
    
    # Create figure
    fig = go.Figure()
    
    # Color palette for sensors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a']
    
    # MEMORY OPTIMIZATION: Plot every 2nd sensor to reduce traces
    sensor_step = 2
    for sensor_idx in range(0, fft_mags.shape[1], sensor_step):
        fft_mags_reduced = fft_mags[freq_indices, sensor_idx]
        fig.add_trace(go.Scatter(
            x=freqs_reduced,
            y=fft_mags_reduced,
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
        title=f"{bearing_name} - FFT Spectrum (Optimized)",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        height=300,  # Reduced height
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

# Memory usage warning
st.info("🔧 **Memory Optimized Mode**: Reduced animation frames and plot points for better performance on cloud platforms.")

# Bearing visualization section
st.subheader("🔄 Animated Bearing Visualization")

# Create two columns for side-by-side bearing plots
col_bearing_a, col_bearing_b = st.columns(2)

with col_bearing_a:
    st.markdown("### Bearing A (19 rolling elements)")
    bearing_a_fig = create_memory_optimized_bearing_plot(
        data['bearing_a'], "Bearing A", freq_selected, time_idx, animation_fps
    )
    st.plotly_chart(bearing_a_fig, use_container_width=True)

with col_bearing_b:
    st.markdown("### Bearing B (21 rolling elements)")
    bearing_b_fig = create_memory_optimized_bearing_plot(
        data['bearing_b'], "Bearing B", freq_selected, time_idx, animation_fps
    )
    st.plotly_chart(bearing_b_fig, use_container_width=True)

st.markdown("---")

# FFT spectrum section
st.subheader("📊 FFT Spectrum Analysis")

# Create two columns for side-by-side FFT plots
col_fft_a, col_fft_b = st.columns(2)

with col_fft_a:
    fft_a_fig = create_memory_optimized_fft_plot(data['bearing_a'], "Bearing A", time_idx, freq_selected)
    st.plotly_chart(fft_a_fig, use_container_width=True)

with col_fft_b:
    fft_b_fig = create_memory_optimized_fft_plot(data['bearing_b'], "Bearing B", time_idx, freq_selected)
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
st.markdown("*Dashboard created for dual bearing fiber sensor FFT analysis - Memory Optimized*") 