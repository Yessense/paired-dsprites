import sys

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from torch.utils.data import DataLoader

sys.path.append("..")
from pytorch_lightning.loggers import WandbLogger

from src.dataset.dataset import Dsprites, PairedDspritesDataset
from src.model.scene_vae import DspritesVAE
import pytorch_lightning as pl
from argparse import ArgumentParser

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_DATASET_PATH = './dataset/data/dsprite_train.npz'
DEFAULT_PAIRED_TRAIN_PATH = './dataset/data/paired_train.npz'
DEFAULT_PAIRED_TEST_PATH = './dataset/data/paired_test.npz'
DEFAULT_LOGGER_DIR_PATH = './'
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------

parser = ArgumentParser()

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--logger_dir", type=str, default=DEFAULT_LOGGER_DIR_PATH)
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
program_parser.add_argument("--paired_train_path", type=str, default=DEFAULT_PAIRED_TRAIN_PATH)
program_parser.add_argument("--paired_test_path", type=str, default=DEFAULT_PAIRED_TEST_PATH)

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=4)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--grad_clip", type=float, default=0.0)

# Add model specific args
parser = DspritesVAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------

wandb_logger = WandbLogger(project='paired-dsprites', log_model=args.log_model, save_dir=args.logger_dir, l)

# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

train_dataset = PairedDspritesDataset(dsprites_path=args.dataset_path,
                                      paired_dsprites_path=args.paired_train_path)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)

test_dataset = PairedDspritesDataset(dsprites_path=args.dataset_path,
                                     paired_dsprites_path=args.paired_test_path)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)
autoencoder = DspritesVAE(**dict_args)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Total loss'

# early stop
patience = 5
early_stop_callback = EarlyStopping(monitor=monitor, patience=patience)

# checkpoint
save_top_k = 2
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

# stohastic weight averaging
swa_lrs = 1e-2
swa = StochasticWeightAveraging(swa_lrs=swa_lrs)

callbacks = [
    checkpoint_callback,
    # swa,
    # early_stop_callback,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------

# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None
gpus = [0]
log_every_n_steps = 200

# trainer
trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger,
                     gradient_clip_val=args.grad_clip,
                     log_every_n_steps=log_every_n_steps)

if not len(args.from_checkpoint):
    args.from_checkpoint = None
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=test_loader, ckpt_path=args.from_checkpoint)
