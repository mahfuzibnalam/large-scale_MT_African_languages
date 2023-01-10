data_bin=$1
save_dir=$2
out=$3

python generate.py $data_bin \
    --path $save_dir/checkpoint_best.pt \
    --model-overrides "{'pretrained_deltalm_checkpoint': '/home/mahfuz/Research/MachineTranslation/Data/DeltaLM/models/multi/pre-trained/deltalm-base.pt'}" \
    --batch-size 16 \
    --beam 5 \
    --remove-bpe=sentencepiece \
> $out.mess

grep ^H $out.mess | cut -f3 >> $out.out
grep ^T $out.mess | cut -f2 >> $out.tgt
grep ^S $out.mess | cut -f2 >> $out.src

sacrebleu $out.out -i $out.tgt -m bleu
#comet-score -s $out.src -t $out.out -r $out.tgt
#python examples/chrF++.py -R $out.tgt -H $out.out

