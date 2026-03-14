import shlex
import subprocess

seeds = []
with open("../RNG-SEEDS.txt", "r") as file:
    seeds = list(map(int, file.read().split()))[0:10]
# Define the list of commands
commands = [
    f"python simulation_mlca.py --domain {domain} --qinit 10 --qround 4 --qmax {qmax} --seed {seed}"
    for seed in seeds
    for domain in ["GSVM", "LSVM", "MRVM", "SRVM"]
    for qmax in [10 * i for i in range(1, 11)]
]
print(commands, len(commands))
exit()

# Note: on Windows, "sleep" and "ls" might not be directly available as executables.
# You may need to use their Windows equivalents like "timeout" and "dir".

print("Script started.")

# Loop through each command and execute it
for command in commands:
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
        print(f"Successfully executed: {' '.join(command)}")
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
