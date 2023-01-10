data_bin=$1
save_dir=$2
PRETRAINED_MODEL=$3

python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_base \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 3e-05 \
    --warmup-updates 2500 \
    --max-tokens 512 \
    --max-update 80000 \
    --save-interval-updates 20000 \
    --update-freq 1 \
    --no-epoch-checkpoints \
    --seed 222 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test

# bash examples/train_data.sh ~/Research/MachineTranslation/Data/DeltaLM/raw/binarized.en-lg ~/Research/MachineTranslation/Data/DeltaLM/models/en-lg/checkpoints ~/Research/MachineTranslation/Data/DeltaLM/models/multi/pre-trained/deltalm-base.pt

