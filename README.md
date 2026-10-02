# MONARCHS validation scripts

Scripts used to generate some of the plots from our paper, *A Model for Antarctic Ice Shelf Hydrology and Stability (MONARCHS v1.0)* (https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3247/). This compares our model output to 
data observed via satellite observations in *Moussavi et al. (2020)* (https://www.mdpi.com/2072-4292/12/1/134) and *Discherl et al. (2021)* (doi).

The main script used to generate the plots for each season can be found in `monarchs_val/run/run_peaks.py`. 

The scripts used to load scenes from the satellite observations can be found in `load/validation_data.py`. 
An example of how to aggregate scenes over a specific temporal window and bounding box can be found in the 
`if __name__ == '__main__':` block at the bottom of that script, in case it is useful.

You will need to point any hard-coded paths to where your GVI/AntarcticLakes data obtained from those papers mentioned above live. I've been running with everything in a `data/` folder at the project root level, and invoking all the scripts from that root level e.g.

```python
python run/run_peaks.py
```

or

```python
python load/validation_data.py
```

so if you do the same and haven't messed about with the default file structure that the data comes with the scripts should work out-of-the-box with no edits needed.