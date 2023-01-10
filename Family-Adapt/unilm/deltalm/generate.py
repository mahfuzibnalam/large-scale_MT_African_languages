import deltalm
import sys

from fairseq_cli.generate import cli_main

if __name__ == "__main__":
    # sys.argv.extend(
    #     [
    #         '/home/mahfuz/Research/MachineTranslation/DeltaLM/raw/binarized.afr-eng_test', 
    #         '--path', '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/test_dataloader/chekpoints/checkpoint_best.pt', 
    #         '--model-overrides', 
    #         "{'pretrained_deltalm_checkpoint': '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/multi/pre-trained/deltalm-base.pt',\
    #         'source_lang': 'af',\
    #         'target_lang': 'en'}",
    #         '--batch-size', '2', 
    #         '--beam', '5', 
    #         '--remove-bpe=sentencepiece',
    #     ])
    cli_main()