import shlex
import subprocess

from tqdm import tqdm

seeds = []
with open("../RNG-SEEDS.txt", "r") as file:
    seeds = list(map(int, file.read().split()))[0:10]
# Define the list of commands

commands = [
    f"python sim_mlca.py --domain {domain} --q {q} --seed {seed} --acquisition uUB_model"
    for seed in seeds
    for domain, q in [("LSVM", 0.90), ("SRVM", 0.75), ("MRVM", 0.90)]
]

# Note: on Windows, "sleep" and "ls" might not be directly available as executables.
# You may need to use their Windows equivalents like "timeout" and "dir".

print("Script started.")

# Loop through each command and execute it
for command in tqdm(commands):
    command = command.split(" ")
    try:
        # subprocess.run waits for the command to finish.
        # 'check=True' will raise an exception if the command fails (exits with a non-zero status).
        # You can use 'capture_output=True' if you need to store the output.
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # print(f"Successfully executed: {' '.join(command)}")
        # Optional: print output
        # print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}")
        print("Error output:", e.stderr)
        # Stop execution if a command fails
        break
    except FileNotFoundError:
        print(
            f"Error: Command not found. Check if the executable is in your PATH. Command: {' '.join(command)}"
        )
        break

print("Script finished.")
