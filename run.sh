python run.py \
  --device 'cpu' \
  --landscape custom \
  --alg botorch \
  --name 'GPmufacnet-botorch-1ADQ' \
  --num_rounds 40 \
  --net GPmufacnet \
  --ensemble_size 1 \
  --out-dir ./result \
  --fitness-data ./unify-length/1ADQ_A.csv \
  --sequence-column 'CDR3' \
  --fitness-column 'Energy' \
  --invert-score


