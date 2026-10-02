"""
Compare the model with the observations for a given date. Observation data is 
loaded within a date range, so that we can account for temporal/spatial sampling and
data quality/cloud issues.
"""

from dataclasses import dataclass
import numpy as np
from rasterio.transform import Affine
from rasterio.transform import rowcol 

from .load.monarchs_data import TimestepToCompare
from .load.validation_data import Scene, nearest_scene, read_lake_mask


@dataclass
class ObsComparison:
    """
    Comparison of one observational dataset with one model date. 
    """

    scene: Scene
    lake: np.ndarray
    transform: Affine
    coverage: float  # % lake, sampled at the model cell centres
    coverage_coarsened: float  # % lake, after putting the observation onto the model grid
    coarsened: np.ndarray  # the observation on the model grid
    offset_days: int  # days between the observation and the model date


@dataclass
class DateComparison:
    """
    The model and each observation set on one date.
    """

    entry: TimestepToCompare
    label: str  # used in the output file names, e.g. "2019_2020"
    model_mask: np.ndarray  # 1 lake, 0 no lake, NaN outside the valid cells
    model_coverage: float
    obs: dict[str, ObsComparison]


def compare_date(setup, entry, scenes):
    """
    Compare the model with the observations for a given date. 

    Parameters
    ----------
    setup: Setup
        The setup info loaded in from the main function arguments
    entry: TimestepToCompare
        The model timestep to compare against
    scenes: dict[str, Scene]
        The nearest scene for each observation dataset, keyed by dataset name
    """
    model_fields = setup.model_fields
    lake_presence = model_fields.lake_presence[entry.model_timestep]
    obs = {}
    for name, found in scenes.items():
        scene = nearest_scene(found, entry.model_date)
        # mask out anything like rocks and clouds that aren't lakes
        mask = read_lake_mask(scene, model_fields.bbox)
        grid_args = (model_fields.x, model_fields.y, setup.domain)
        coarsened, coverage_coarsened = coarsen_validation_to_model_grid(
            mask.lake,
            mask.transform,
            *grid_args,
            threshold=setup.coarsen_threshold,
        )
        obs[name] = ObsComparison(
            scene=scene,
            lake=mask.lake,
            transform=mask.transform,
            coverage=sample_validation_coverage_on_model_grid(
                mask.lake, mask.transform, *grid_args
            ),
            coverage_coarsened=coverage_coarsened,
            coarsened=coarsened,
            offset_days=abs((scene.date - entry.model_date).days),
        )
    # calculate lake coverage from the model, accounting only valid cells in 
    # the domain
    n_valid = int(setup.domain.sum())
    coverage = float((lake_presence & setup.domain).sum()) / n_valid * 100.0
    return DateComparison(
        entry=entry,
        label=entry.selection_label.replace(" austral summer", "")
        .replace("-", "_")
        .replace(" ", "_"),
        model_mask=np.where(model_fields.valid_mask, lake_presence.astype(float), np.nan),
        model_coverage=coverage,
        obs=obs,
    )

def sample_validation_coverage_on_model_grid(
    lake_mask, tile_transform, model_x, model_y, domain
):
    """
    Calculate lake coverage from the observations, using the model cell centres as the sampling points.
    domain here is the valid cells in the model grid that we are comparing against

    Parameters
    ----------
    lake_mask: np.ndarray   
        The observed lake mask, 1 for lake, 0 for no lake, NaN for outside the observation
    tile_transform: Affine
        The affine transform for the observation tile, so we can convert model cell centres to observation pixel
        coordinates
    model_x: np.ndarray
        The x coordinates of the model cell centres in EPSG:3031
    model_y: np.ndarray
        The y coordinates of the model cell centres in EPSG:3031
    domain: np.ndarray
        The valid model cells that we are comparing against, 1 for valid, 0 for invalid
    """
    n_valid = int(domain.sum())
    obs_rows, obs_cols = rowcol(tile_transform, model_x[domain], model_y[domain])
    obs_rows = np.asarray(obs_rows, dtype=int)
    obs_cols = np.asarray(obs_cols, dtype=int)

    n_obs_rows, n_obs_cols = lake_mask.shape
    # determine which model cell centres are within the bounds of the observation and use these
    # to calculate the number of lakes
    in_bounds = (obs_rows >= 0) & (obs_rows < n_obs_rows) & (obs_cols >= 0) & (obs_cols < n_obs_cols)
    n_lake = int(lake_mask[obs_rows[in_bounds], obs_cols[in_bounds]].sum())
    return n_lake / n_valid * 100.0


def coarsen_validation_to_model_grid(
    lake_mask,
    tile_transform,
    model_x,
    model_y,
    domain,
    threshold=0.5,
):
    """
    Put observations onto the model grid, so we can plot them on the same grid and calculate differences.
    Threshold determines the number of pixels within a model grid cell that need to be lakes in order for that cell
    to be considered a lake when plotting. 

    Parameters
    ----------
    lake_mask: np.ndarray
        The observed lake mask, 1 for lake, 0 for no lake, NaN for outside the observation
    tile_transform: Affine
        The affine transform for the observation tile, so we can convert model cell centres to observation pixel
        coordinates
    model_x: np.ndarray
        The x coordinates of the model cell centres in EPSG:3031
    model_y: np.ndarray
        The y coordinates of the model cell centres in EPSG:3031
    domain: np.ndarray
        The valid model cells that we are comparing against, 1 for valid, 0 for invalid
    threshold: float
        The fraction of observed pixels within a model cell that need to be lakes in order
        for that cell to be considered a lake when plotting. Default is 0.5.
    """
    n_domain = int(domain.sum())
    obs_pixel_m = abs(tile_transform.a)  # metres per obs pixel

    # Estimate model cell spacing w.r.t. the obs pixel size using adjacent cells in the 
    # model domain 
    pair_rows, pair_cols = np.where(domain[:-1, :] & domain[1:, :])
    dx = model_x[pair_rows + 1, pair_cols] - model_x[pair_rows, pair_cols]
    dy = model_y[pair_rows + 1, pair_cols] - model_y[pair_rows, pair_cols]
    model_spacing_m = float(np.median(np.sqrt(dx ** 2 + dy ** 2)))
    
    # half_window is the number of obs pixels to each side of the model cell centre to include in the square
    half_window = max(1, round(model_spacing_m / obs_pixel_m / 2))
    model_rows, model_cols = np.where(domain)
    obs_rows, obs_cols = rowcol(
        tile_transform, model_x[model_rows, model_cols], model_y[model_rows, model_cols]
    )
    obs_rows = np.asarray(obs_rows, dtype=int)
    obs_cols = np.asarray(obs_cols, dtype=int)

    n_obs_rows, n_obs_cols = lake_mask.shape
    # array of coarsened observations on the model grid 
    coarsened = np.full(model_x.shape, np.nan)

    # For each model cell, look at the corresponding observation pixels and determine if it's a lake
    for model_row, model_col, obs_row, obs_col in zip(model_rows, model_cols, obs_rows, obs_cols):
        row_start = max(0, obs_row - half_window)
        row_end = min(n_obs_rows, obs_row + half_window + 1)
        col_start = max(0, obs_col - half_window)
        col_end = min(n_obs_cols, obs_col + half_window + 1)

        if row_end <= row_start or col_end <= col_start:
            coarsened[model_row, model_col] = 0.0
            continue

        window = lake_mask[row_start:row_end, col_start:col_end]
        lake_fraction = int(window.sum()) / window.size
        coarsened[model_row, model_col] = 1.0 if lake_fraction >= threshold else 0.0

    # Estimate the coverage percentage of cells that have lakes in the domain
    coverage_pct = float(np.sum(coarsened[domain] == 1.0)) / n_domain * 100.0

    return coarsened, coverage_pct
