#!/bin/bash
#SBATCH --job-name=tunnel
#SBATCH --gpus=1            # this also allocates 72 CPU cores and 115GB memory
#SBATCH --time=6:00:00
#SBATCH --output=tunnel_%j.out

# Start named VS Code tunnel for remote connection to compute node
~/opt/vscode_cli/code tunnel --name "u5aa_tunnel"


