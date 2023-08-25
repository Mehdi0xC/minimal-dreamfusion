# Use this script to reproduce results in the README.md file
# Default config is fetched from config/default.yaml
# Overriding configs can be passed as arguments to the script

# Reproduce experiment #1
python anc_sampling.py default output_dir=outputs/exp1 CFG=6

# Reproduce experiment #2
python sds_sampling.py default output_dir=outputs/exp2 CFG=100

# Reproduce experiment #3
python sds_sampling.py default output_dir=outputs/exp3-1 CFG=2 
python sds_sampling.py default output_dir=outputs/exp3-2 CFG=6
python sds_sampling.py default output_dir=outputs/exp3-3 CFG=10
python sds_sampling.py default output_dir=outputs/exp3-4 CFG=25
python sds_sampling.py default output_dir=outputs/exp3-5 CFG=50
python sds_sampling.py default output_dir=outputs/exp3-6 CFG=100

# Reproduce experiment #4
python sds_sampling.py default output_dir=outputs/exp4 CFG=50 prior=True max_noise=750