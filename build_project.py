import subprocess
import os

def build_project(log_name="compile_log.txt"):
    """
    Compiles all Python files in the current working directory recursively.
    Logs the results to a file and stops if any error is found.
    """

    # Get the current absolute path for logging clarity
    cwd = os.getcwd()

    bash_script_content = f"""#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

LOG_FILE="{cwd}/{log_name}"

echo "=== Starting Compilation in $PWD ===" | tee "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "------------------------------------" | tee -a "$LOG_FILE"

# Find all .py files and attempt to compile them
# Using -print0 and read -d '' to safely handle filenames with spaces
find . -name "*.py" -print0 | while read -d '' -r pyfile; do
    echo "Checking: $pyfile" | tee -a "$LOG_FILE"

    # Run the compiler. 2>> redirects error output to the log file.
    python -m py_compile "$pyfile" 2>> "$LOG_FILE" || {{
        echo -e "\nERROR: Compilation failed for $pyfile" | tee -a "$LOG_FILE"
        echo "Build process terminated." | tee -a "$LOG_FILE"
        exit 1
    }}
done

echo "------------------------------------" | tee -a "$LOG_FILE"
echo "Success: All files compiled." | tee -a "$LOG_FILE"
echo "Build completed! Log saved to $LOG_FILE"
"""

    # Write the temporary bash script
    script_path = "run_compile.sh"
    with open(script_path, "w") as f:
        f.write(bash_script_content)

    # Make executable
    subprocess.run(["chmod", "+x", script_path])

    try:
        # Execute the compilation
        result = subprocess.run(["bash", script_path], check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Compilation Failed:\n{e.stderr}")
        # Even if it fails, the partial log will be in your directory
        print(f"Check {log_name} for the specific syntax error.")
    finally:
        # Cleanup the temporary shell script
        if os.path.exists(script_path):
            os.remove(script_path)
