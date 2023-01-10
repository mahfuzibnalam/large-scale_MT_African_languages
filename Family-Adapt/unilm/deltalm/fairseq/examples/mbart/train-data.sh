DATA=$1
OUTPUT=$2
langs=$3

fairseq-train $DATA \
  --save-dir $OUTPUT \
  --encoder-normalize-before \
  --decoder-normalize-before \
  --layernorm-embedding \
  --langs $langs \
  --arch mbart_large \
  --task multilingual_denoising \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.2 \
  --optimizer adam \
  --adam-eps 1e-06 \
  --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay \
  --lr 0.00003 \
  --warmup-updates 250 \
  --total-num-update 4000 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --weight-decay 0.0 \
  --max-tokens 512 \
  --max-update 4000 \
  --save-interval-updates 1000 \
  --keep-best-checkpoints 5 \
  --seed 222 \
  --replace-length 1 \
  --multilang-sampling-alpha 0.7 \
  --add-lang-token \
  --mask 0.35 \
  --permute-sentences 1 \
  --rotate 0.0 \
  --ddp-backend legacy_ddp
