# Pre-processing
Both train and test source sentence has to be preporcessed by adding this token "<<b>src</b>-<b>tgt</b>>" at the begining. For each language pair <b>src</b> is the source_language's ISO code and <b>tgt</b> is the target_language's ISO code. Then merge together all language pairs source sentences into one file and target sentences into another file. 
```
Without preprocessing (en to af) source sentence:
Hen went away to mend her husband's pants. The next morning, as usual, Hen was on her way to Eagle.
With preprocessing (en to af) source sentence:
<en-af> Hen went away to mend her husband's pants. The next morning, as usual, Hen was on her way to Eagle.
```
Important: The ISO codes everywhere for this experiments have to be same for a language.

# Sub-word Tokenization
Use <b>models/spm/spm.model</b> to create the tokenized version using sentencepiece.

# Binarized File for Training
<pre>
train_input_dir=The directory where tokenized Training files are
valid_input_dir=The directory where tokenized Validation files are
dict_file=<b>models/vocab/dict.txt</b>
output_dir=The binarized Training and Validation data file's directory

python preprocess.py  \
    --trainpref $train_input_dir/train.spm \
    --validpref $valid_input_dir/valid.spm \
    --source-lang src \
    --target-lang tgt \
    --destdir $output_dir\
    --srcdict $dict_file \
    --tgtdict $dict_file \
    --workers 80 
</pre>

# Training
```
data_bin=The binarized Training and Validation data file's directory
save_dir=The directory where the model will be saved
PRETRAINED_MODEL=The directory where the pre-trained model is saved e.g., models/WMT22_langadapt.pt

python train.py $data_bin \
    --save-dir $save_dir \
    --arch deltalm_large \
    --pretrained-deltalm-checkpoint $PRETRAINED_MODEL \
    --share-all-embeddings \
    --max-source-positions 512 \
    --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr 1e-04 \
    --warmup-updates 2000 \
    --max-update 80000 \
    --max-epoch 8 \
    --max-tokens 1536 \
    --save-interval-updates 4000 \
    --no-epoch-checkpoints \
    --update-freq 1 \
    --seed 222 \
    --log-format simple \
    --skip-invalid-size-inputs-valid-test \
    --re-tuned \
    --adapter \
    --adapter-enc-dim 64 \
    --adapter-dec-dim 64 \
    --adapter-groups (Put all the Family names here, comma-separated e.g., indo-european,afro-asiatic,nilo-saharan,bantu,volta-niger,senegambian) \
    --adapter-families (Put all the Sub-Family names here, comma-separated e.g., germanic,french,hausa,amharic,cushitic,luo,wolof,fula,igboid,yoruboid,northeast-bantu,nguni-tsonga,sotho-tswana,bangi,shona,nyasa,umbundu,sabi) \
    --adapter-langs (Put all the Language ISO 639‑3 names here, comma-separated e.g., af,am,en,fr,fuv,ha,ig,kam,rw,lg,ln,luo,nso,ny,om,sn,so,ss,sw,tn,ts,umb,wo,xh,yo,zu,bem) \
    --adapter-hierarchies (Put all the hierachies in the same order as the languages above here, semicolon-separated for hierachies and comma-separated inside hierarchy e.g., indo-european,germanic,af;afro-asiatic,amharic,am;indo-european,germanic,en;indo-european,french,fr;senegambian,fula,fuv;afro-asiatic,hausa,ha;volta-niger,igboid,ig;bantu,northeast-bantu,kam;bantu,northeast-bantu,rw;bantu,northeast-bantu,lg;bantu,bangi,ln;nilo-saharan,luo,luo;bantu,sotho-tswana,nso;bantu,nyasa,ny;afro-asiatic,cushitic,om;bantu,shona,sn;afro-asiatic,cushitic,so;bantu,nguni-tsonga,ss;bantu,northeast-bantu,sw;bantu,sotho-tswana,tn;bantu,nguni-tsonga,ts;bantu,umbundu,umb;senegambian,wolof,wo;bantu,nguni-tsonga,xh;volta-niger,yoruboid,yo;bantu,nguni-tsonga,zu;bantu,sabi,bem \
    --adapter-unfreeze (Put which type of adapters you want to train as comma-separated e.g., groups,families,langs) \
    --ddp-backend=legacy_ddp
```

# Binarized File for Testing
<pre>
test_input_dir=The directory where tokenized Testing files are
dict_file=<b>models/vocab/dict.txt</b>
output_dir=The binarized Testing data file's directory
src=The language code which should be in the above --adapter-langs.
tgt=The language code which should be in the above --adapter-langs.

python preprocess.py  \
    --testpref $test_input_dir/test.spm \
    --source-lang $src \
    --target-lang $tgt \
    --destdir $output_dir \
    --srcdict $dict_file \
    --tgtdict $dict_file \
    --workers 80
</pre>

# Testing 
Testing is done between only one language pair.
```
data_bin=The binarized Testing data file's directory
model_dir=The directory where the model has been be saved
PRETRAINED_MODEL=The directory where the pre-trained model to create the above model is saved e.g., models/WMT22_langadapt.pt
src=The language code which should be in the above --adapter-langs.
tgt=The language code which should be in the above --adapter-langs.
python generate.py $data_bin \
    --path $model_dir \
    --model-overrides "{'pretrained_deltalm_checkpoint': '$PRETRAINED_MODEL', 'source_lang': '$src', 'target_lang': '$tgt'}" \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe=sentencepiece \
> output.mess

grep ^H output.mess | sort -V | cut -f3 > output
```
