python run.py \
  --device 'cuda:0' \
  --landscape custom \
  --alg botorch \
  --name 'esm-botorch-1ADQ' \
  --num_rounds 40 \
  --net esm1b \
  --ensemble_size 1 \
  --out-dir /home/tianyu/code/proximal-exploration-active-learning/result \
  --fitness-data /home/tianyu/code/biodrug/unify-length/1ADQ_A.csv \
  --sequence-column 'CDR3' \
  --fitness-column 'Energy' \
  --invert-score


