python run.py \
  --device 'cpu' \
  --landscape custom \
  --alg pex \
  --name 'mufacnet-pex-1ADQ' \
  --num_rounds 40 \
  --net mufacnet \
  --ensemble_size 1 \
  --out-dir ./result \
  --fitness-data ./unify-length/1ADQ_A.csv \
  --sequence-column 'CDR3' \
  --fitness-column 'Energy' \
  --invert-score


