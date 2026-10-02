"""
Read the MONARCHS output, and select dates to compare with
the satellite observations. Just handles loading lake depth and
coverage at the moment, but could be extended to load other fields
by extending ModelFields
"""

from dataclasses import dataclass
from datetime import date, timedelta
import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


@dataclass
class ModelFields:
    """
    Info needed by the plotting/validation script about the MONARCHS output.
    """

    dates: list[date]  # date of each output time step
    lake_presence: (
        np.ndarray
    )  # lake mask used internally by the model, shape (time, rows, cols)
    lake_depth: np.ndarray  # lake depth in metres, shape (time, rows, cols)
    valid_mask: np.ndarray  # cells that we actually do physics on, shape (rows, cols)
    lat: np.ndarray
    lon: np.ndarray
    x: (
        np.ndarray
    )  # cell centres in metres for EPSG:3031, shape (rows, cols), along with y
    y: np.ndarray  # as above
    bbox: BaseGeometry  # bounding box in EPSG:3031 of the model grid
    coverage_pct: (
        np.ndarray
    )  # % of valid cells with a lake, over the whole grid over all model timesteps


@dataclass
class TimestepToCompare:
    """
    Info on a model timestep that we want to then cross-compare with the validation data
    """

    model_timestep: int  # model timestep index
    model_date: date  # date of the model timestep
    selection_label: str  # label for the plot and CSV, e.g. "2014-2015 austral summer". Auto-generated
    # depending on whether it is found as a peak for a given season or set as a fixed date


def get_seasonal_peak_entries(model_fields):
    """
    Find the model timestep with the highest lake coverage in each southern hemisphere summer season
    """
    # label the peaks by year
    by_season = {}
    for timestep, model_date in enumerate(model_fields.dates):
        # nov, dec
        if model_date.month in (11, 12):
            season_year = model_date.year
        # jan, feb
        elif model_date.month in (1, 2):
            season_year = model_date.year - 1
        else:
            continue
        by_season.setdefault(season_year, []).append(timestep)

    entries = []
    for season_year in sorted(by_season.keys()):
        timesteps = by_season[season_year]
        # find index with most coverage in this season
        peak_timestep = timesteps[int(np.argmax(model_fields.coverage_pct[timesteps]))]
        entries.append(
            TimestepToCompare(
                model_timestep=peak_timestep,
                model_date=model_fields.dates[peak_timestep],
                selection_label=f"{season_year}-{season_year + 1} austral summer",
            )
        )
    return entries


def get_entry_for_fixed_date(model_fields, fixed_date):
    """
    If selecting a specific date for comparison, find the corresponding timestep in the model output
    file and make that the thing we are comparing against
    """
    offsets = [abs((model_date - fixed_date).days) for model_date in model_fields.dates]
    # index is just the one with the smallest gap in days from the selected date
    nearest_timestep = int(np.argmin(offsets))
    return TimestepToCompare(
        model_timestep=nearest_timestep,
        model_date=model_fields.dates[nearest_timestep],
        selection_label=f"fixed date {fixed_date.isoformat()}",
    )


def load_model_fields(
    model_output_path, progress_path, threshold, start_date_text, save_freq
):
    """
    Read lake depth and cell coordinates from the model output, and which cells are actually on an ice shelf
    (i.e. valid cells) from the progress dumpfile, as we dont save the mask in the output by default.
    Threshold is the size above which we have a lake, typically 0.1 m (which is what determines the
    mask internally in the model).

    Returns a ModelFields object with all of the relevant data the plotting/validation script needs.

    Parameters
    ----------
    model_output_path: Path
        Path to the MONARCHS model output file (netCDF)
    progress_path: Path
        Path to the MONARCHS progress/dump file (netCDF)
    threshold: float
        The size above which we have a lake, typically 0.1 m (which is what determines the mask internally in the model)
    start_date_text: str
        The start date of the model output, in ISO format (YYYY-MM-DD)
    save_freq: int
        The number of days between model outputs (5 default which is what I used for the MONARCHS runs)
    """
    with Dataset(model_output_path) as model_ds, Dataset(progress_path) as progress_ds:
        lake_depth = np.asarray(model_ds.variables["lake_depth"][:])
        valid_mask = np.asarray(progress_ds.variables["valid_cell"][:]).astype(bool)
        lat = np.asarray(model_ds.variables["lat"][:])
        lon = np.asarray(model_ds.variables["lon"][:])

    # check in case we've loaded the progress file instead of the model output file
    # (in which case lake depth is 2D)
    if lake_depth.ndim != 3:
        raise ValueError(
            f"Expected lake_depth shape (time, x, y), got {lake_depth.shape}. "
            f"Was the progress file loaded by mistake?"
        )

    # find number of cells with a lake
    lake_presence = (lake_depth > threshold) & valid_mask[None, :, :]
    #
    total_lakes = lake_presence.sum(axis=(1, 2))
    valid_count = int(valid_mask.sum())
    # then coverage is total_lakes / total_valid_cells * 100
    coverage_pct = total_lakes / valid_count * 100.0
    # convert selected date to DateTime object
    start = date.fromisoformat(start_date_text)
    # generate the list of dates for each timesteps
    dates = [start + timedelta(days=i * save_freq) for i in range(lake_depth.shape[0])]
    # convert lat/lon to EPSG:3031 for plotting and bounding box
    x, y = Transformer.from_crs(4326, 3031, always_xy=True).transform(lon, lat)

    return ModelFields(
        dates=dates,
        lake_presence=lake_presence,
        lake_depth=lake_depth,
        valid_mask=valid_mask,
        lat=lat,
        lon=lon,
        x=x,
        y=y,
        bbox=box(np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)),
        coverage_pct=coverage_pct,
    )


def select_area(model_fields, area_rows, area_cols):
    """
    Select the area to compare: the valid model cells in rows `area_rows` and columns `area_cols`, and the
    rectangle around them in EPSG:3031, which sets the map extent and the area searched for GVI depths

    Parameters
    ----------
    model_fields: ModelFields   
        The model fields object containing the model grid and other relevant data
    area_rows: tuple[int, int]
        The rows of the model grid to compare (start, end)
    area_cols: tuple[int, int]
        The columns of the model grid to compare (start, end)
    """
    rows, cols = slice(*area_rows), slice(*area_cols)
    domain = np.zeros_like(model_fields.valid_mask)
    domain[rows, cols] = model_fields.valid_mask[rows, cols]
    x, y = model_fields.x[rows, cols], model_fields.y[rows, cols]
    area = box(np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y))
    return domain, area
