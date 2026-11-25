# AerCloudDiff

AerCloudDiff predicts 24 hourly Himawari B13 brightness-temperature fields from satellite history and WRF-Chem forecasts.

| Model | Aerosol inputs |
| --- | --- |
| noAer | None |
| 2DAer | AOD550 and 28 aerosol column summaries |
| 3DAer | 2DAer inputs plus nine aerosol groups at six native model levels |

## Setup

Use Python 3.11 or later. Run commands from the source directory.

```bash
python -m pip install -r requirements.txt
SAMPLES=/path/to/hdf5_samples
CHECKPOINTS=/path/to/checkpoints
```

Use daily 12 UTC HDF5 samples on the 149 by 191 grid, named `sample_YYYYMMDD12.h5`. The 2025 split is January to October for training (304 files), November for validation (30), and December for evaluation (31). See the [HDF5 reader](src/data/dataset_b13cascade.py) and [aerosol cache builder](scripts/build_aerosol_cache.py) for input fields. The [normalization assets](assets/normalization) contain the WRF-cloud and 3DAer aerosol statistics.

Download the two case samples, predictions, animations and manuscript source data as `AerCloudDiff-data_20261123.zip` [data DOI to be added]. Extract it next to this directory. The examples are under `../AerCloudDiff-data/cases/2025121112/` and `../AerCloudDiff-data/cases/2025121612/`; each contains `sample_YYYYMMDDHH.h5`, three `<model>_YYYYMMDDHH.npz` files and `forecast.gif`. The data dictionary describes the fields and original data sources. Use the examples and reader to check the format of additional prepared samples.

## Training

Use four GPUs on one node. Run the 3DAer GPU stage in a Slurm allocation with four A100 GPUs. 3DAer freezes the 2DAer backbone and trains only the vertical aerosol branch. Supply a compatible 2DAer EMA checkpoint for 3DAer initialization and an initial predicted-history cache for 2DAer training.

To build your own initial history cache, train the supplied auxiliary history model and run the history builder on the training split:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train_b13cascade.py --config configs/support/2DAer_history.yaml \
  --sample-dir "$SAMPLES" --output-dir runs/2DAer_history
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/build_b13_predicted_history_cache.py \
  --source-config configs/support/2DAer_history.yaml \
  --source-checkpoint runs/2DAer_history/checkpoints/epoch_100.pt \
  --sample-dir "$SAMPLES" --output-dir runs/2DAer_history_initial \
  --split train --sampling-steps 100 --weights ema --seed 42
```

The auxiliary model supplies predicted-history inputs for 2DAer training.

```bash
python scripts/train.py noAer --sample-dir "$SAMPLES" --stop-after-epoch 45
python scripts/train.py 2DAer --sample-dir "$SAMPLES" \
  --history-cache runs/2DAer_history_initial --stop-after-epoch 50

python scripts/workflow_3daer.py prepare --sample-dir "$SAMPLES" \
  --baseline runs/2DAer/checkpoints/epoch_035.pt
python scripts/workflow_3daer.py run --sample-dir "$SAMPLES" \
  --baseline runs/2DAer/checkpoints/epoch_035.pt \
  --stop-after-epoch 45 --evaluation-epoch 30
```

The 80-epoch setting defines the learning-rate schedule. `--stop-after-epoch` controls the execution length. Training records are 45, 50 and 45 epochs; evaluation uses EMA weights from epochs 30, 35 and 30. The training wrapper saves these numbered checkpoints. It also saves the checkpoint selected by validation as `best.pt` or `best_trained.pt`.

noAer and 2DAer warm up the learning rate once per epoch, while 3DAer updates it at each optimizer step; all three reach the peak learning rate of `2e-5` by the end of the first five epochs. The 3DAer training dataset retains all `304 × 5 × 12 = 18,240` window-patch entries, or 1,520 optimizer steps per epoch with a global batch size of 12. Observation masks control loss and metric denominators; a window with no valid observations remains in the sampling schedule and contributes zero observation loss. Frames with no valid verification observations remain in the forecast output and have unscored metrics.

The 3DAer workflow checks the supplied 2DAer EMA structure before initialization. Preparation uses the supplied aerosol normalization parameters through `build_aerosol_cache.py --normalization-file`; calling the builder without this option fits statistics from training samples.

## Inference

Evaluate the December samples with EMA checkpoints and 100-step DDIM on the full grid.

```bash
for MODEL in noAer 2DAer; do
  if [ "$MODEL" = noAer ]; then EPOCH=030; else EPOCH=035; fi
  python scripts/eval_b13cascade.py --config "configs/$MODEL.yaml" \
    --checkpoint "runs/$MODEL/checkpoints/epoch_$EPOCH.pt" --sample-dir "$SAMPLES" \
    --output-dir "predictions/$MODEL" --split test --max-files 31 --full-grid \
    --weights ema --sampling-steps 100 --sampling-seed 42 \
    --metrics-min-lon 80 --save-predictions
done

python scripts/eval_3daer.py --config reports/3DAer/prepared_config.yaml \
  --checkpoint runs/3DAer/checkpoints/epoch_030.pt --split test \
  --output-dir predictions/3DAer
```

Prediction files use `sample_YYYYMMDDHH_pred.npz` in each model's output directory. The distributed cases use `<model>_YYYYMMDDHH.npz` to identify models within a shared directory.

noAer and 3DAer use `sample_keyed` sampling; 2DAer uses `legacy_global`. The optional HDF5 attribute `sampling_seeds_42` stores five integer seeds for base seed 42, ensemble member 0 and forecast windows 0-4. The evaluator uses these values for the five forecast windows. Other settings derive seeds from the current sample name.

`report_3daer.py` summarizes training runs using frame-wise metrics. The data package's source-data tables and error moments provide the area-weighted manuscript statistics and paired resampling indices.

The stored predictions can be inspected and scored directly.