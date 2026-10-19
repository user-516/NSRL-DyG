# Dynamic Graph Denoising by Breaking the Noise Domino Effect
### Data Preprocess

As we built our work upon NSRL-DyG, most of the used original dynamic graph datasets come from [Towards Better Evaluation for Dynamic Link Prediction](https://github.com/yule-BUAA/TGEDataset), which can be downloaded [here](https://github.com/yule-BUAA/TGEDataset). 
### Training

By default the datasets are read from `./data`. Either place `ml_<name>.csv` and the
matching `.npy` files there, or point `--data_root` (or the `NSRL_DATA_ROOT`
environment variable) at the directory that holds them.

Train the model for the AUC evaluation

```bash
python trainAUC.py --data_root /path/to/datasets
```

### License

All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
