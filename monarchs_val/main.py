"""
Compare MONARCHS lakes with the GVI and AntarcticLakes observations. This is done via looking for
when the model has peak lake coverage, and comparing those with the nearest satellite observations.

- `peak_comparison_<season>.png`: lake maps of MONARCHS and of each observation set.
- `peak_comparison_coarsened_t<threshold>_<season>.png`: the same with the observations put onto the model grid.
- `peak_histograms_<season>.png`: model and GVI lake depths, and the lake coverage of each dataset.

Each figure gets some metadata in JSON format so we can try and reproduce them based on the 
settings and input files used.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
from shapely.geometry.base import BaseGeometry

from .compare import compare_date
from .load.monarchs_data import (
    ModelFields,
    get_entry_for_fixed_date,
    get_seasonal_peak_entries,
    load_model_fields,
    select_area,
)
from .load.validation_data import (
    Scene,
    choose_gvi_depths,
    find_antarcticlakes_scenes,
    find_gvi_scenes,
)
from .plot.plotting import plot_coarsened_maps, plot_depth_histograms, plot_lake_maps

DATASET_CHOICES = ("model", "gvi", "antarcticlakes")


@dataclass
class Setup:
    """
    Setup info loaded in from the main function arguments 
    """
    outdir: Path
    datasets: list[str]
    model_fields: ModelFields
    domain: np.ndarray  # the valid model cells compared, which count towards coverage
    area: BaseGeometry  # rectangle around them: the map extent, and where GVI depths are searched
    depth_scenes: list[Scene]  # GVI scenes over area
    lake_threshold: float
    temporal_window_days: int
    coarsen_threshold: float
    metadata: dict  # recorded in the JSON file saved with each figure


def run_validation(
    model_output,
    progress,
    gvi_dir,
    antarcticlakes_dir,
    outdir,
    area_rows,
    area_cols,
    datasets=None,
    start_date="2014-01-01",
    save_every_timesteps=5,
    lake_threshold=0.1,
    temporal_window_days=15,
    fixed_date=None,
    coarsen_threshold=0.5,
):
    """
    Compare the model's lakes with the observations and write the figures and JSON files to `outdir`.

    Parameters
    ----------
    model_output : str
        Path to the MONARCHS model output file (netCDF)
    progress : str
        Path to the MONARCHS progress/dump file (netCDF)
    gvi_dir : str
        Path to the GVI dataset directory
    antarcticlakes_dir : str
        Path to the AntarcticLakes dataset directory
    outdir : str
        Path to the output directory where the figures will be saved
    area_rows : tuple[int, int]
        The rows of the model grid to compare (start, end)
    area_cols : tuple[int, int]     
        The columns of the model grid to compare (start, end) 
    datasets : list[str], optional
        The datasets to compare: "model", "gvi", "antarcticlakes"
    start_date : str, optional
        The date of the first model output, in YYYY-MM-DD format (default: "2014-01-01")
    save_every_timesteps : int, optional
        Number of days between model outputs (5 default which is what I used for the MONARCHS runs)
    lake_threshold : float, optional
        Depth threshold above which a model cell is considered to have a lake (default: 0.1 m)
    temporal_window_days : int, optional
        The number of days before and after the model date to search for GVI depths (default: 15)
    fixed_date : str, optional
        If provided, only compare the model and observations on this date (in YYYY-MM-DD format)
    coarsen_threshold : float, optional
        The threshold for coarsening the observations to the model grid (default: 0.5)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_output, progress = Path(model_output), Path(progress)
    gvi_dir, antarcticlakes_dir = Path(gvi_dir), Path(antarcticlakes_dir)
    datasets = [name for name in DATASET_CHOICES if datasets is None or name in datasets]

    print("Loading model data...")
    model_fields = load_model_fields(
        model_output,
        progress,
        threshold=lake_threshold,
        start_date_text=start_date,
        save_freq=save_every_timesteps,
    )
    domain, area = select_area(model_fields, area_rows, area_cols)
    xmin, ymin, xmax, ymax = area.bounds

    print("Finding satellite scenes...")
    scenes = {}
    if "gvi" in datasets:
        scenes["gvi"] = find_gvi_scenes(gvi_dir, model_fields.bbox)
    if "antarcticlakes" in datasets:
        scenes["antarcticlakes"] = find_antarcticlakes_scenes(antarcticlakes_dir, model_fields.bbox)
    for name, found in scenes.items():
        if not found:
            raise ValueError(f"No {name} scenes overlap the model grid. Check the {name} directory.")
        print(f"  {name}: {len(found)} scenes")

    if fixed_date:
        entries = [
            get_entry_for_fixed_date(model_fields, date.fromisoformat(fixed_date))
        ]
    else:
        entries = get_seasonal_peak_entries(model_fields)

    # put all the input arguments into a config object - used to make the metadata file later
    setup = Setup(
        outdir=outdir,
        datasets=datasets,
        model_fields=model_fields,
        domain=domain,
        area=area,
        depth_scenes=find_gvi_scenes(gvi_dir, area),
        lake_threshold=lake_threshold,
        temporal_window_days=temporal_window_days,
        coarsen_threshold=coarsen_threshold,
        metadata=dict(
            input_paths={
                "model_output": str(model_output),
                "progress": str(progress),
                "gvi_dir": str(gvi_dir),
                "antarcticlakes_dir": str(antarcticlakes_dir),
            },
            parameters={
                "start_date": start_date,
                "save_every_timesteps": save_every_timesteps,
                "lake_threshold": lake_threshold,
                "temporal_window_days": temporal_window_days,
                "fixed_date": fixed_date,
                "datasets": datasets,
                "coarsen_threshold": coarsen_threshold,
            },
            domain={
                "area_rows": area_rows,
                "area_cols": area_cols,
                "extent_xmin": xmin,
                "extent_xmax": xmax,
                "extent_ymin": ymin,
                "extent_ymax": ymax,
            },
        ),
    )

    for entry in entries:
        print(
            f"\n--- Processing: {entry.selection_label} (model date: {entry.model_date}) ---"
        )
        comparison = compare_date(setup, entry, scenes)
        plot_lake_maps(setup, comparison)
        plot_coarsened_maps(setup, comparison)
        gvi_depths = choose_gvi_depths(
            setup.depth_scenes, setup.area, entry.model_date, setup.temporal_window_days
        )
        plot_depth_histograms(setup, comparison, gvi_depths)
