import os
import glob
import subprocess
from collections import defaultdict

def main():
    # Dictionary to store results: results[model][method][length] = accuracy
    # Using defaultdict for easier nested assignment
    results = {
        "LLaDA": defaultdict(dict),
        "Dream": defaultdict(dict)
    }

    # Find all jsonl sample files
    files = glob.glob("evals_results/**/samples_*.jsonl", recursive=True)
    if not files:
        print("No samples_*.jsonl files found in evals_results/")
        return

    print("Running post-processing on found log files. This may take a moment...")

    for f in files:
        parts = f.split(os.sep)
        
        # Determine model and correct postprocessing script based on directory structure
        if "humaneval" == parts[1]:  # evals_results/humaneval/...
            model = "LLaDA"
            post_script_dir = "llada"
        elif "HumanEval-Dream" == parts[1]: # evals_results/HumanEval-Dream/...
            model = "Dream"
            post_script_dir = "dream"
        else:
            continue

        # Extract sequence length and method
        try:
            len_part = next(p for p in parts if p.startswith("len"))
            length = len_part.replace("len", "")
            len_idx = parts.index(len_part)
            method = parts[len_idx + 1]
        except StopIteration:
            continue

        print(f"Processing {model} | Method: {method:20} | Length: {length}...")
        
        abs_path = os.path.abspath(f)
        
        # Run the script inside the respective directory so `import sanitize` works
        result = subprocess.run(
            ["python", "postprocess_code.py", abs_path],
            cwd=post_script_dir,
            capture_output=True,
            text=True
        )

        output = result.stdout.strip()
        
        try:
            # We expect the final output line to be the float accuracy
            acc = float(output.split('\n')[-1])
            results[model][method][length] = acc
        except ValueError:
            print(f"[!] Warning: Failed to parse output for {f}")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")

    # Generate Markdown Tables
    print("\n" + "="*50)
    print("ALL POST-PROCESSED RESULTS")
    print("="*50 + "\n")

    methods_order = ["baseline", "prefix-cache", "parallel", "prefix-cache-parallel", "dual-cache-parallel"]

    for model in ["LLaDA", "Dream"]:
        print(f"### HumanEval - {model} (Gen length 256/512)\n")
        print("| Method | Accuracy @256 | Accuracy @512 |")
        print("|---|---:|---:|")
        
        model_results = results[model]
        
        # Ensure we capture all methods found in case some differ from expected
        found_methods = set(model_results.keys())
        display_methods = [m for m in methods_order if m in found_methods]
        display_methods += [m for m in found_methods if m not in display_methods]

        for method in display_methods:
            acc_256 = model_results[method].get("256")
            acc_512 = model_results[method].get("512")
            
            str_256 = f"{acc_256:.4f}" if acc_256 is not None else "N/A"
            str_512 = f"{acc_512:.4f}" if acc_512 is not None else "N/A"
            
            print(f"| {method.capitalize()} | {str_256} | {str_512} |")
        
        print("\n")

if __name__ == "__main__":
    main()
