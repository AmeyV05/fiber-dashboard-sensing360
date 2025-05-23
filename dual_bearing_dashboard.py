import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit
st.set_page_config(layout="wide", page_title="Dual Bearing FFT Visualization")

# --- ULTRA LIGHTWEIGHT SAMPLE DATA ---
@st.cache_data(ttl=3600, max_entries=1)
def generate_minimal_demo_data():
    """Generate ultra-lightweight demo data for Streamlit Cloud"""
    # Minimal parameters for memory efficiency
    sampling_frequency = 100  # Much lower sampling rate
    duration = 2  # Very short duration
    shaft_speed_rpm = 450
    fundamental_frequency = 7.5
    
    # Minimal time vector
    num_samples = int(sampling_frequency * duration)
    time_vector = np.linspace(0, duration, num_samples)
    
    # Minimal frequency vector (only up to 50 Hz)
    freqs = np.linspace(0, 50, 50)
    
    # Bearing configurations - simplified
    bearing_a_angles = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
    bearing_b_angles = np.array([15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345, 375])
    
    # Generate very simple FFT data
    def generate_simple_fft(angles):
        num_sensors = len(angles)
        fft_mags = np.zeros((len(freqs), num_sensors))
        fft_phases = np.zeros((len(freqs), num_sensors))
        
        for sensor_idx, angle in enumerate(angles):
            # Simple peaks at fundamental frequencies
            for freq_mult in [1, 2, 3]:
                target_freq = fundamental_frequency * freq_mult
                freq_idx = int(target_freq)  # Simple indexing
                if freq_idx < len(freqs):
                    amplitude = 1.0 / freq_mult
                    fft_mags[freq_idx, sensor_idx] = amplitude
                    fft_phases[freq_idx, sensor_idx] = angle
        
        return fft_mags, fft_phases
    
    bearing_a_fft_mags, bearing_a_fft_phases = generate_simple_fft(bearing_a_angles)
    bearing_b_fft_mags, bearing_b_fft_phases = generate_simple_fft(bearing_b_angles)
    
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

# --- SIMPLIFIED DATA LOADING ---
@st.cache_data(ttl=3600, max_entries=1)
def load_data():
    """Load data with memory optimization"""
    # Always use demo data for Streamlit Cloud
    st.info("🔬 **Demo Mode**: Using minimal synthetic data optimized for Streamlit Cloud")
    return generate_minimal_demo_data()

# Load data
data = load_data()
time_vector = data['time_vector']
freqs = data['freqs']
fundamental_freq = data['fundamental_frequency']
time_formatted = pd.to_datetime(time_vector, unit='s')
num_times = len(time_vector)

# --- PAGE HEADER ---
st.title("🔧 Dual Bearing Fiber Sensor Dashboard")
st.markdown(f"**RPM:** {data['shaft_speed_rpm']} | **Freq:** {fundamental_freq:.1f} Hz | **Rate:** {data['sampling_frequency']} Hz")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ Controls")

# Simple frequency selection
freq_options = [7.5, 15.0, 22.5, 30.0]
freq_selected = st.sidebar.selectbox("Frequency (Hz)", freq_options, index=0)

# Time index
time_idx = st.sidebar.slider("Time", 0, num_times-1, 0)

# --- SIMPLE BEARING VISUALIZATION ---
def create_simple_bearing_plot(bearing_data, bearing_name):
    """Create a simple static bearing plot"""
    angles = bearing_data['angles']
    
    # Get simple magnitude values
    freq_idx = int(freq_selected / 50 * len(freqs))  # Simple frequency mapping
    if freq_idx < bearing_data['fft_mags'].shape[0]:
        magnitudes = bearing_data['fft_mags'][freq_idx, :]
    else:
        magnitudes = np.ones(len(angles)) * 0.5
    
    # Normalize magnitudes
    max_mag = max(magnitudes) if max(magnitudes) > 0 else 1.0
    normalized_mags = 0.5 + (magnitudes / max_mag) * 0.5
    
    fig = go.Figure()
    
    # Add bearing circle
    circle_theta = np.linspace(0, 360, 37)  # Reduced points
    fig.add_trace(go.Scatterpolar(
        r=[1]*37,
        theta=circle_theta,
        mode='lines',
        line=dict(color='gray', width=1),
        showlegend=False
    ))
    
    # Add sensor points
    fig.add_trace(go.Scatterpolar(
        r=normalized_mags,
        theta=angles,
        mode='markers+lines',
        marker=dict(size=8, color='red'),
        line=dict(color='blue', width=2),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"{bearing_name}",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.5], showticklabels=False),
            angularaxis=dict(direction="clockwise", rotation=90)
        ),
        height=300,
        margin=dict(t=50, l=20, r=20, b=20)
    )
    
    return fig

# --- SIMPLE FFT PLOT ---
def create_simple_fft_plot(bearing_data, bearing_name):
    """Create a simple FFT plot"""
    fft_mags = bearing_data['fft_mags']
    
    # Average magnitude across sensors to reduce memory
    avg_magnitude = np.mean(fft_mags, axis=1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs,
        y=avg_magnitude,
        mode='lines',
        name='Average',
        line=dict(color='blue', width=2)
    ))
    
    # Mark selected frequency
    fig.add_vline(x=freq_selected, line=dict(color="red", width=2, dash="dash"))
    
    fig.update_layout(
        title=f"{bearing_name} - FFT",
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        height=200,
        margin=dict(t=40, l=20, r=20, b=20)
    )
    
    return fig

# --- MAIN LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Bearing A")
    fig_a = create_simple_bearing_plot(data['bearing_a'], "Bearing A")
    st.plotly_chart(fig_a, use_container_width=True)
    
    fft_a = create_simple_fft_plot(data['bearing_a'], "Bearing A")
    st.plotly_chart(fft_a, use_container_width=True)

with col2:
    st.subheader("Bearing B") 
    fig_b = create_simple_bearing_plot(data['bearing_b'], "Bearing B")
    st.plotly_chart(fig_b, use_container_width=True)
    
    fft_b = create_simple_fft_plot(data['bearing_b'], "Bearing B")
    st.plotly_chart(fft_b, use_container_width=True)

# --- SIMPLE INFO ---
st.markdown("---")
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Frequency", f"{freq_selected} Hz")
with col_info2:
    st.metric("Time", f"{time_idx}/{num_times-1}")
with col_info3:
    st.metric("Duration", f"{len(time_vector)/data['sampling_frequency']:.1f}s")

# --- TECHNICAL INFO ---
with st.expander("ℹ️ System Info"):
    st.write("**Memory Optimized Version for Streamlit Cloud**")
    st.write(f"- Bearing A: {len(data['bearing_a']['angles'])} sensors, {data['bearing_a']['rolling_elements']} elements")
    st.write(f"- Bearing B: {len(data['bearing_b']['angles'])} sensors, {data['bearing_b']['rolling_elements']} elements")
    st.write(f"- Data size: {data['bearing_a']['fft_mags'].nbytes + data['bearing_b']['fft_mags'].nbytes} bytes")

st.markdown("---")
st.markdown("*Lightweight demo for dual bearing analysis*") 