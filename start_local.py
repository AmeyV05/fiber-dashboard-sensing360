#!/usr/bin/env python3
"""
Local Development Startup Script

This script is for local development only and helps start the dashboard locally.
For Streamlit Cloud deployment, use dual_bearing_dashboard.py directly.
"""

import os
import sys
import subprocess

def check_processed_data():
    """Check if processed data file exists"""
    if not os.path.exists('processed_dual_bearing_data.h5'):
        print("❌ Processed data file not found!")
        print("Please run the preprocessing script first:")
        print("python preprocess_dual_bearing_data.py")
        return False
    return True

def main():
    """Main function for local development"""
    print("🔧 Dual Bearing Fiber Sensor Dashboard - Local Development")
    print("=" * 60)
    
    if not check_processed_data():
        sys.exit(1)
    
    print("✅ Data file found!")
    print("🚀 Starting dashboard locally...")
    print("📊 Dashboard features:")
    print("  - Side-by-side animated bearing visualizations")
    print("  - Real-time FFT spectrum analysis")
    print("  - Interactive frequency selection")
    print("  - Time series navigation")
    print("\n🌐 Dashboard will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Run streamlit directly
        subprocess.run(['streamlit', 'run', 'dual_bearing_dashboard.py'], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except FileNotFoundError:
        print("\n❌ Streamlit not found! Please install it with:")
        print("pip install streamlit")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")

if __name__ == "__main__":
    main() 