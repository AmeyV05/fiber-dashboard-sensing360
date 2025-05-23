# 🔧 Dual Bearing Fiber Sensor FFT Visualization Dashboard

This project provides a comprehensive visualization dashboard for analyzing fiber optic sensor data from dual bearings using FFT analysis and animated real-time displays.

## 📋 Project Overview

The dashboard visualizes fiber sensor data from two bearings (A and B) mounted on the same shaft, each equipped with 13 equidistantly positioned sensors around the outer ring. The system provides:

- **Animated bearing visualizations** showing real-time fiber sensor responses
- **FFT spectrum analysis** for frequency domain analysis
- **Interactive controls** for frequency selection and time navigation
- **Side-by-side comparison** of both bearings

### System Specifications

- **Bearing A**: 13 sensors, 19 rolling elements
- **Bearing B**: 13 sensors, 21 rolling elements
- **Shaft Speed**: 450 RPM (7.5 Hz fundamental frequency)
- **Sampling Rate**: 2000 Hz
- **Data Format**: Fiber optic wavelength measurements

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv fiber_venv

# Activate virtual environment
source fiber_venv/bin/activate  # On Mac/Linux
# OR
fiber_venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing

Run the preprocessing script to convert the .mat file to optimized HDF5 format:

```bash
python preprocess_dual_bearing_data.py
```

This script:
- Loads the raw .mat file containing fiber sensor data
- Creates equidistant time series (2000 Hz)
- Performs FFT analysis on all sensors
- Saves processed data in HDF5 format with proper structure
- Generates validation plots

### 3. Launch Dashboard

#### Local Development
```bash
# Method 1: Direct launch
streamlit run dual_bearing_dashboard.py

# Method 2: Using local helper script
python start_local.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

#### Streamlit Cloud Deployment

For deployment on Streamlit Cloud:

1. **Push to GitHub**: Ensure all files are committed and pushed to your GitHub repository

2. **Access Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)

3. **Deploy App**: 
   - Connect your GitHub account
   - Select your repository: `fiber-dashboard-sensing360`
   - Set main file path: `dual_bearing_dashboard.py`
   - Set branch: `main`

4. **Configuration Files**: The following files are included for Streamlit Cloud:
   - `requirements.txt` - Python dependencies
   - `packages.txt` - System dependencies 
   - `.streamlit/config.toml` - Streamlit configuration

5. **Data File**: Ensure `processed_dual_bearing_data.h5` is included in your repository (or run preprocessing on the cloud)

**Important**: Do not use `run_dashboard.py` as the main file for Streamlit Cloud deployment, as it's designed for local development only.

## 📊 Dashboard Features

### Main Visualization Components

1. **Animated Bearing Plots**
   - Real-time sinusoidal animation based on selected frequency
   - Individual sensor positions and magnitudes
   - Smooth interpolated curves showing bearing deformation
   - Play/Pause controls for animation

2. **FFT Spectrum Analysis**
   - Full frequency spectrum for all sensors
   - Selectable frequency highlighting
   - Fundamental frequency markers
   - Individual sensor traces

3. **Interactive Controls**
   - Frequency selection (multiples of fundamental frequency)
   - Time index navigation
   - Animation speed control
   - Real-time parameter updates

### Sidebar Controls

- **Frequency Selection**: Choose from fundamental frequency multiples (1x to 16x)
- **Animation FPS**: Adjust animation speed (2-10 FPS)
- **Time Index**: Navigate through the time series data
- **Current Time Display**: Shows selected timestamp

### Information Panels

- **Technical Information**: Sensor counts, rolling elements, and angles
- **Data Information**: Time duration, frequency resolution, and data shapes
- **System Metrics**: Real-time display of selected parameters

## 📁 Clean Project Structure

```
fiber-dashboard-sensing360/
├── 📋 Core Files
│   ├── requirements.txt                     # Python dependencies
│   ├── README.md                           # Complete documentation
│   └── project_structure.md               # Project organization guide
│
├── 🔧 Data Processing
│   ├── preprocess_dual_bearing_data.py    # Main preprocessing script
│   └── optics11_recording_*.mat           # Raw fiber sensor data
│
├── 📊 Dashboard Application
│   ├── dual_bearing_dashboard.py          # Main Streamlit dashboard
│   └── run_dashboard.py                   # Dashboard runner script
│
├── 📈 Generated Data & Outputs
│   ├── processed_dual_bearing_data.h5     # Processed HDF5 data
│   └── validation_plots/                  # Validation outputs
│       ├── fft_validation_plot.png
│       └── sensor_layout.png
│
└── 🐍 Environment
    └── fiber_venv/                         # Python virtual environment
```

**Benefits of Refactored Structure:**
- ✅ **Simplified**: Only essential files remain
- ✅ **Clear Purpose**: Each file has a distinct, well-defined role
- ✅ **Easy Maintenance**: Logical organization for future updates
- ✅ **Reproducible**: Virtual environment with pinned dependencies

## 🔧 Technical Details

### Data Processing Pipeline

1. **Raw Data Loading**: Loads .mat file containing wavelength data and timestamps
2. **Time Series Alignment**: Creates equidistant time series at 2000 Hz
3. **Data Interpolation**: Interpolates sensor data to new time grid
4. **FFT Analysis**: Computes magnitude and phase spectra for each sensor
5. **Data Organization**: Structures data by bearing with proper metadata

### Sensor Configuration

#### Bearing A Sensors
- **Sensor Indices**: 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
- **Angles (degrees)**: 67.0, 94.7, 122.4, 150.1, 177.8, 205.5, 233.2, 260.8, 288.5, 316.2, 343.9, 371.6, 399.3

#### Bearing B Sensors
- **Sensor Indices**: 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14
- **Angles (degrees)**: 282.0, 254.3, 226.6, 198.9, 171.2, 143.5, 115.8, 88.2, 60.5, 32.8, 5.1, -22.6, -50.3

### Frequency Analysis

The dashboard focuses on bearing-specific frequencies:
- **Fundamental**: 7.5 Hz (450 RPM)
- **Harmonics**: 2x, 3x, 4x, 5x, 8x, 12x, 16x fundamental
- **Window Size**: ±0.2 Hz for peak detection

## 🛠️ Dependencies

Core packages (see `requirements.txt` for exact versions):
- `streamlit` - Web dashboard framework
- `plotly` - Interactive plotting
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `pandas` - Data manipulation
- `h5py` - HDF5 file handling
- `matplotlib` - Additional plotting support

## 📈 Usage Examples

### Analyzing Bearing Health

1. **Select Fundamental Frequency** (7.5 Hz) to observe basic shaft rotation
2. **Compare Bearing Responses** by examining side-by-side animations
3. **Check Harmonic Content** by selecting higher frequency multiples
4. **Navigate Time Series** to observe temporal changes

### Frequency Domain Analysis

1. **Examine FFT Spectra** to identify dominant frequencies
2. **Look for Peaks** at expected bearing fault frequencies
3. **Compare Sensor Responses** within each bearing
4. **Monitor Frequency Evolution** across time

### Animation Analysis

1. **Play Animations** to visualize dynamic behavior
2. **Adjust Speed** for detailed observation
3. **Pause at Specific Phases** for detailed analysis
4. **Compare Phase Relationships** between bearings

## 🚨 Troubleshooting

### Common Issues

1. **Missing Data File**
   ```
   Error: processed_dual_bearing_data.h5 not found
   Solution: Run preprocessing script first
   ```

2. **Import Errors**
   ```
   Error: Module not found
   Solution: Activate virtual environment and install requirements
   ```

3. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Reduce data size or increase system memory
   ```

### Performance Tips

- Use lower animation FPS for smoother performance
- Focus on lower frequency ranges (0-200 Hz) for faster rendering
- Close other browser tabs to free memory
- Use caching for repeated operations

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is intended for research and educational purposes. Please refer to your institution's policies for data usage and sharing.

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the validation plots in `validation_plots/`
- Examine the console output for error messages
- Verify all dependencies are correctly installed

---

*Dashboard created for dual bearing fiber sensor FFT analysis - Research Project 2024* 