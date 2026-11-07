# Leakage-aware heart disease prediction benchmark (UCI, n = 920)

## Environment
Python 3.13.18. Install with: pip install -r requirements-pinned.txt

## Data
Download the four processed.*.data files from the UCI Heart Disease repository
(Cleveland, Hungarian, Switzerland, VA Long Beach) into data/uci/.

## Reproduce the tables
Table 4 (pooled benchmark):
  python run_benchmark.py --data-dir data/uci --sites all --repeats 5 --n-iter 12 --calibrate sigmoid --out results/full_all_sites

Table 3 (leakage arms, run one learner at a time):
  python run_core_arms.py --learners LR --repeats 10
  python run_core_arms.py --learners GB --repeats 10
  python run_core_arms.py --learners RF --repeats 10

Mechanism arms D and E:
  python run_arm_d.py --learner RF --repeats 10
  python run_arm_e.py --learner RF --repeats 10

Parent count:
  python run_mechanism_count.py

Seed stability:
  python run_seed_stability.py --learner GB --seeds 1,2,3 --repeats 3
  python run_seed_stability.py --learner RF --seeds 1,2,3 --repeats 3

Table 5 (leave-one-site-out):
  python run_loso_saved.py --n-iter 12
  python loso_bootstrap.py

Table S2 (calibration):
  python loso_calibration.py
  python make_table_s2.py

## Notes
- Random seeds are fixed (CV seed 42; SMOTE seeds 1 to 3 in the stability check).
- Synthetic-data smoke tests (--demo) are for code testing only and are not results.
- Duplicate audit: one exact duplicate each in Hungarian and VA, retained.
- The MLP is in the code but excluded from the reported comparison (did not converge within 2000 iterations).
