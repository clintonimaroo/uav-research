#!/bin/bash
#SBATCH --job-name=uav_comparison
#SBATCH --output=comparison_%j.out
#SBATCH --error=comparison_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=RM-shared

# Change to the navigation directory
cd /ocean/projects/cis250165p/cimaro/uav-research/navigation

# Load any necessary modules (if needed)
# module load python/3.8  # Uncomment if you want to use a newer Python version

# Run the comparison script
python3 compare_methods.py --episodes 50 --grid 50 --max_steps 200

