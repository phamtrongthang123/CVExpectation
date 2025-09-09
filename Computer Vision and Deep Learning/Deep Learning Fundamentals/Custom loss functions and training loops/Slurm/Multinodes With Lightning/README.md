simply
```bash
sbatch train_apptainer.sh
```
btw, if you want to load ckpt after it being converted to a single file, try load as state_dict then fit normally, because they won't save optimizer states and many meta info.
