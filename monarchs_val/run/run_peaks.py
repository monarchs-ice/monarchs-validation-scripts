"""
Run the peak comparison code to find where the peaks are in the model output and
plot up lakes at the corresponding times
"""

from monarchs_val import run_validation

run_validation(
    model_output="model_output/model_output_flow_1.0.nc",
    progress="model_output/progress_flow_1.0.nc",
    gvi_dir="validation/GVI",
    antarcticlakes_dir="validation/AntarcticLakes/George_VI",
    outdir="output/peaks",
    start_date="2014-01-01",  # date of the first model output
    save_every_timesteps=5,  # days between model outputs
    area_rows=(0, 61),  # compare model grid rows 0 to 60 and columns 0 to 60,
    area_cols=(0, 61),  # the north-west of the 100 x 100 grid
    # threshold for how much lake coverage within a model cell shows up as an obs 
    # lake so we can compare on the same grid
    coarsen_threshold=0.5,  
)
