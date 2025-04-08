# -------------- main.py --------------
# Main script that runs the full pipeline step-by-step
import subprocess
import os
import time

def create_directories():
    """Create necessary directories for outputs"""
    directories = ["3rd Assignment/figures"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def run_script(script_path):
    """Run a Python script and handle errors"""
    print(f"\n{'='*50}")
    print(f"Running: {script_path}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"\n✅ Successfully completed in {time.time() - start_time:.2f} seconds")
        return True
    else:
        print(f"\n❌ Error running {script_path}:")
        print(result.stderr)
        return False

def main():
    """Run the full analysis pipeline"""
    print("Starting Fast Growth Prediction Pipeline")
    print("-" * 50)
    
    # Create necessary directories
    create_directories()
    
    # Define scripts to run in order
    scripts = [
        "3rd Assignment/Code/1.data_prep.py",
        "3rd Assignment/Code/2.modeling.py",
        "3rd Assignment/Code/3.classification.py",
        "3rd Assignment/Code/4.industry_comparison.py"
    ]
    
    # Run each script in sequence
    success = True
    for script in scripts:
        if not os.path.exists(script):
            print(f"\n❌ Script not found: {script}")
            success = False
            continue
            
        script_success = run_script(script)
        if not script_success:
            success = False
            print("\n⚠️ Continuing with next script despite errors...")
    
    if success:
        print("\n🎉 All steps completed successfully!")
    else:
        print("\n⚠️ Pipeline completed with some errors. Check the output above for details.")

if __name__ == "__main__":
    main()