# SBATCH --job-name

# SBATCH --nodes=1
# SBATCH --ntasks=20
# SBATCH --cpus-per-task=32
# SBATCH --mem=16GB
# SBATCH --time=24:00:00
# SBATCH --output=slurm_%A_%a.out


 glacier region = 17
 get_glacier_list(glac_nums)
for glacier_num in get_glacier_list
        srun --tasks=1 --cpus-per-task=32 --mem=16GB --time=24:00:00 --output=slurm_%A_%a.out python -u run_calibration_MB_FA_Paralle.py glacier_num(i) & wait