python run.py \
  --device 'cpu' \
  --landscape custom \
  --alg antbo \
  --name 'antbo2-cnn-1ADQ' \
  --num_rounds 40 \
  --net cnn \
  --ensemble_size 3 \
  --out-dir ./result \
  --fitness-data ./unify-length/1ADQ_A.csv \
  --sequence-column 'CDR3' \
  --fitness-column 'Energy' \
  --invert-score


