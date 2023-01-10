input_dir=$1
SPM_MODEL=$2
src=$3
tgt=$4

mkdir -p $input_dir/spm.$src-$tgt

for split in train valid test ;
do
    for lang in $src $tgt ;
    do
        cat $input_dir/tokenized.$src-$tgt/$split.$lang | spm_encode --model=$SPM_MODEL --output_format=piece > $input_dir/spm.$src-$tgt/$split.$lang;
    done;
done;

# bash examples/spm_data.sh ~/Research/Dataset/MachineTranslation/Data/DeltaLM/raw ~/Research/MachineTranslation/Data/DeltaLM/models/multi/spm/spm.model en lg