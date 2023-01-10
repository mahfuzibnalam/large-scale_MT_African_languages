input_dir=$1
dict_file=$2
src=$3
tgt=$4

python preprocess.py  \
    --testpref $input_dir/spm.$src-$tgt/test \
    --source-lang $src --target-lang $tgt \
    --destdir $input_dir/binarized.$src-$tgt \
    --srcdict $dict_file \
    --tgtdict $dict_file \
    --workers 40\
    --trainpref $input_dir/spm.$src-$tgt/train \
    --validpref $input_dir/spm.$src-$tgt/valid 
