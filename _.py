import os
import subprocess
result = subprocess.run(
    [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe", "--version"],
    capture_output=True, text=True
)
print(result.stdout)
# print(os.environ["PATH"])