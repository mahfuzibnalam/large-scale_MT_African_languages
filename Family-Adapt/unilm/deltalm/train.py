import deltalm
import sys

from fairseq_cli.train import cli_main


if __name__ == "__main__":
    # sys.argv.extend(
    #     [
    #         '/home/mahfuz/Research/MachineTranslation/DeltaLM/raw/binarized.all/',
    #         '--save-dir', '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/test_dataloader/chekpoints',
    #         '--arch', 'deltalm_base',
    #         '--pretrained-deltalm-checkpoint', '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/multi/pre-trained/deltalm-base.pt',
    #         '--share-all-embeddings',
    #         '--max-source-positions', '512',
    #         '--max-target-positions', '512',
    #         '--criterion', 'label_smoothed_cross_entropy',
    #         '--label-smoothing', '0.1',
    #         '--optimizer', 'adam',
    #         '--adam-betas', '(0.9, 0.98)',
    #         '--lr-scheduler', 'inverse_sqrt',
    #         '--lr', '3e-05',
    #         '--warmup-updates', '100',
    #         '--max-tokens', '512',
    #         '--max-update', '20000',
    #         '--save-interval-updates', '200',
    #         '--update-freq', '1',
    #         '--no-epoch-checkpoints',
    #         '--seed', '222',
    #         '--log-format', 'simple',
    #         '--skip-invalid-size-inputs-valid-test',
    #         # '--re-tuned',
    #         '--adapter',
    #         '--adapter-enc-dim', '64',
    #         '--adapter-dec-dim', '64',
    #         '--adapter-groups', 'indo-european',
    #         '--adapter-families', 'germanic',
    #         '--adapter-langs', 'af,en',
    #         '--adapter-hierarchies', 'indo-european,germanic,af;indo-european,germanic,en',
    #         '--adapter-unfreeze', 'families,langs',
    #     ])
    # sys.argv.extend(
    #     [
    #         '/home/mahfuz/Research/MachineTranslation/DeltaLM/raw/binarized.all/', 
    #         '--save-dir', '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/test_dataloader/chekpoints', 
    #         '--arch', 'deltalm_base', 
    #         '--pretrained-deltalm-checkpoint', '/home/mahfuz/Research/MachineTranslation/DeltaLM/models/multi/pre-trained/deltalm-base.pt',
    #         '--share-all-embeddings', 
    #         '--max-source-positions', '512', 
    #         '--max-target-positions', '512', 
    #         '--criterion', 'label_smoothed_cross_entropy', 
    #         '--label-smoothing', '0.1', 
    #         '--optimizer', 'adam', 
    #         '--adam-betas', '(0.9, 0.98)', 
    #         '--lr-scheduler', 'inverse_sqrt', 
    #         '--lr', '3e-05', 
    #         '--warmup-updates', '100', 
    #         '--max-tokens', '512', 
    #         '--max-update', '20000', 
    #         '--save-interval-updates', '200', 
    #         '--update-freq', '1', 
    #         '--no-epoch-checkpoints', 
    #         '--seed', '222', 
    #         '--log-format', 'simple', 
    #         '--skip-invalid-size-inputs-valid-test',
    #     ])
    cli_main()