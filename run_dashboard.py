#!/usr/bin/env python3
"""
Dual Bearing Fiber Sensor Dashboard Runner

This script helps to run the Streamlit dashboard for dual bearing visualization.
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

def check_requirements():
    """Check if virtual environment and requirements are satisfied"""
    if not os.path.exists('fiber_venv'):
        print("❌ Virtual environment not found!")
        print("Please create virtual environment first:")
        print("python -m venv fiber_venv")
        print("source fiber_venv/bin/activate")
        print("pip install -r requirements.txt")
        return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Dual Bearing Dashboard...")
    print("📊 Dashboard features:")
    print("  - Side-by-side animated bearing visualizations")
    print("  - Real-time FFT spectrum analysis")
    print("  - Interactive frequency selection")
    print("  - Time series navigation")
    print("\n🌐 Dashboard will open in your browser automatically")
    print("   If not, go to: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Activate virtual environment and run streamlit
        if os.name == 'nt':  # Windows
            activate_cmd = 'fiber_venv\\Scripts\\activate'
            cmd = f'{activate_cmd} && streamlit run dual_bearing_dashboard.py'
        else:  # Unix/Linux/MacOS
            activate_cmd = 'source fiber_venv/bin/activate'
            cmd = f'{activate_cmd} && streamlit run dual_bearing_dashboard.py'
        
        subprocess.run(cmd, shell=True, check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running dashboard: {e}")
        print("Make sure all requirements are installed correctly")

def main():
    """Main function"""
    print("🔧 Dual Bearing Fiber Sensor Dashboard")
    print("=" * 40)
    
    # Check prerequisites
    if not check_requirements():
        sys.exit(1)
    
    if not check_processed_data():
        sys.exit(1)
    
    print("✅ All prerequisites met!")
    print()
    
    # Run dashboard
    run_dashboard()

if __name__ == "__main__":
    main() 