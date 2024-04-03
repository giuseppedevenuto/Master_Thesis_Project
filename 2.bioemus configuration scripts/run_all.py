import subprocess 
import os

for fname in os.listdir('./config'):
    if '.json' in fname:
        subprocess.run(["./app/run.sh", "config/"+fname])