#!/usr/bin/env python3
"""
Dual Bearing Fiber Sensor Dashboard Runner

This script helps to run the Streamlit dashboard for dual bearing visualization.
"""

import os
import sys
import subprocess
import streamlit.web.bootstrap

def check_processed_data():
    """Check if processed data file exists"""
    if not os.path.exists('processed_dual_bearing_data.h5'):
        print("❌ Processed data file not found!")
        print("Please run the preprocessing script first:")
        print("python preprocess_dual_bearing_data.py")
        return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Dual Bearing Dashboard...")
    
    try:
        # Run streamlit directly without virtual environment activation
        if os.environ.get('IS_STREAMLIT_CLOUD'):
            # On Streamlit Cloud, just import and run the dashboard module
            import dual_bearing_dashboard
        else:
            # Local development - use subprocess
            subprocess.run(['streamlit', 'run', 'dual_bearing_dashboard.py'], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")

def main():
    """Main function"""
    if not check_processed_data():
        sys.exit(1)
    
    run_dashboard()

if __name__ == "__main__":
    main() 