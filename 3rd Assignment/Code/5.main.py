# -------------- main.py --------------
# Main script that runs the full pipeline step-by-step
import subprocess

scripts = [
    "3rd Assignment/Code/1.data_prep.py",
    "3rd Assignment/Code/2.modeling.py",
    "3rd Assignment/Code/3.classification.py",
    "3rd Assignment/Code/4.industry_comparison.py"
]

for script in scripts:
    print(f"\n--- Running: {script} ---")
    subprocess.run(["python", script], check=True)

print("\n✅ All steps completed successfully!")