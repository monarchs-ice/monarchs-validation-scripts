"""
Main model vs obs plotting functions
"""

import cartopy.crs as ccrs
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.plot import plotting_extent
from .metadata import DatasetInfo, record_and_save

# this massively helps running on my mac as otherwise 
# it runs interactively with the macos backend and it locks up
matplotlib.use("Agg")  


# define the CRS - EPSG:3031 with the true scale latitude at -71
# without setting this the coastlines aren't in the right place
EPSG3031 = ccrs.Stereographic(
    central_latitude=-90,
    central_longitude=0,
    true_scale_latitude=-71,
    false_easting=0,
    false_northing=0,
    globe=ccrs.Globe(datum="WGS84", ellipse="WGS84"),
)

# labels and colours for the different datasets for the bar chart and figure titles
TITLES = {"model": "MONARCHS", "gvi": "Moussavi et al. (2020)", "antarcticlakes": "Discherl et al. (2021)"}
BAR_LABELS = {"model": "MONARCHS", "gvi": "Moussavi (2020)", "antarcticlakes": "Discherl (2021)"}
BAR_COLOURS = {"model": "steelblue", "gvi": "coral", "antarcticlakes": "darkorange"}

# colour maps for plotting
# cmap such that non-lake observed pixels are light grey rather than white, and 
# lakes are blue
BINARY_LAKE_CMAP = ListedColormap(["#f2f2f2", "#2166ac"])
BINARY_LAKE_CMAP.set_bad((1.0, 1.0, 1.0, 0.0))
# ensure 0 and 1 are the only values in the colour map 
BINARY_LAKE_NORM = BoundaryNorm([-0.5, 0.5, 1.5], BINARY_LAKE_CMAP.N)


def plot_lake_maps(setup, comparison_data):
    """
    plot panels of lake maps for the model and each observation set, cropped to the area of interest

    Parameters
    ----------
    setup: Setup
        The setup info loaded in from the main function arguments
    comparison_data: DateComparison
        The model and each observation set on one date, with the observations coarsened to the model grid
    """
    shown = []
    panels = []
    for name in setup.datasets:
        if name == "model":
            panels.append((TITLES["model"], comparison_data.model_mask, None))
            # add the model data to the metadata
            shown.append(
                DatasetInfo(
                    dataset="model",
                    path=setup.metadata["input_paths"]["model_output"],
                    date=comparison_data.entry.model_date.isoformat(),
                    offset_days=0,
                )
            )
        else:
            obs = comparison_data.obs[name]
            panels.append((TITLES[name], obs.lake.astype(float), obs.transform))
            # add the obs data to the metadata
            shown.append(
                DatasetInfo(
                    dataset=name,
                    path=str(obs.scene.path),
                    date=obs.scene.date.isoformat(),
                    offset_days=obs.offset_days,
                )
            )
    fig = _map_figure(panels, setup.model_fields, setup.area)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_with_metadata(fig, setup.outdir / f"peak_comparison_{comparison_data.label}.png", setup.metadata, shown)


def plot_coarsened_maps(setup, comparison_data):
    """
    Plot lake coverage maps for the model and observations with the observations coarsened 
    to the model grid so we compare like for like (somewhat)

    Parameters
    ----------
    setup: Setup
        The setup info loaded in from the main function arguments
    comparison_data: DateComparison
        The model and each observation set on one date, with the observations coarsened to the model grid   
    """
    panels = []
    for name in setup.datasets:
        if name == "model":
            title = TITLES["model"]
            coverage, lake_map = comparison_data.model_coverage, comparison_data.model_mask
        else:
            obs = comparison_data.obs[name]
            title = f"{TITLES[name]} — coarsened"
            coverage, lake_map = obs.coverage_coarsened, obs.coarsened
        panels.append((f"{title}\ncoverage: {coverage:.2f}%", lake_map, None))
    fig = _map_figure(panels, setup.model_fields, setup.area)
    threshold = setup.coarsen_threshold
    fig.suptitle(f"Obs coarsened to model grid  |  threshold: fraction ≥ {threshold}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    thresh_str = f"{threshold:g}".replace(".", "p")
    _save_with_metadata(
        fig,
        setup.outdir / f"peak_comparison_coarsened_t{thresh_str}_{comparison_data.label}.png",
        setup.metadata,
    )


def plot_depth_histograms(setup, comparison_data, gvi_depths):
    """
    Plot a histogram of the model vs observed lake depths and bar charts showing lake coverage

    Parameters 
    ----------
    setup: Setup
        The setup info loaded in from the main function arguments
    comparison_data: DateComparison
        The model and each observation set on one date, with the observations coarsened to the model grid
    gvi_depths: tuple[datetime.date, np.ndarray] or None
        The GVI scene with the most lake pixels within the temporal window, 
        and its depths in m, restricted to the area of interest. 
        None if no scene in the window has lake depths
    """
    model_date = comparison_data.entry.model_date
    lake_depth = setup.model_fields.lake_depth[comparison_data.entry.model_timestep]
    fig, (ax_model, ax_gvi, ax_bar) = plt.subplots(1, 3, figsize=(18, 4.5))

    model_lake_depths = lake_depth[setup.domain & (lake_depth > setup.lake_threshold)]
    if model_lake_depths.size > 0:
        _depth_hist(ax_model, model_lake_depths, "steelblue")
    else:
        ax_model.text(0.5, 0.5, "No lake pixels", ha="center", va="center", transform=ax_model.transAxes)
    ax_model.set_title(f"MONARCHS\n{model_date.isoformat()}", fontsize=12)

    if gvi_depths is not None:
        gvi_date, depths = gvi_depths
        _depth_hist(ax_gvi, depths, "coral")
        ax_gvi.set_title(
            f"{TITLES['gvi']}\n{gvi_date.isoformat()} (±{setup.temporal_window_days} d)", fontsize=12
        )
    else:
        ax_gvi.text(
            0.5,
            0.5,
            "No GVI depth data available",
            ha="center",
            va="center",
            transform=ax_gvi.transAxes,
            fontsize=14,
        )
        ax_gvi.set_title(TITLES["gvi"], fontsize=12)

    for ax in (ax_model, ax_gvi):
        ax.set_xlabel("Lake depth (m)", fontsize=12)
        ax.set_ylabel("Number of pixels", fontsize=12)
        ax.grid(True, alpha=0.3)

    names = ["model", *comparison_data.obs]
    coverages = [comparison_data.model_coverage] + [obs.coverage for obs in comparison_data.obs.values()]
    bars = ax_bar.bar(
        [BAR_LABELS[name] for name in names],
        coverages,
        color=[BAR_COLOURS[name] for name in names],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, coverage in zip(bars, coverages):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{coverage:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_bar.set_ylabel("Lake coverage (%)", fontsize=12)
    # add a small offset so the text fits
    ax_bar.set_ylim(0, max(coverages) * 1.25 + 0.5)
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_title("Lake coverage inside area of interest")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_with_metadata(fig, setup.outdir / f"peak_histograms_{comparison_data.label}.png", setup.metadata)


def _map_figure(panels, model_fields, area):
    """
    Create a figure object with subplots for each panel, and plot the data on them.

    Parameters
    ----------
    panels: list[tuple[str, np.ndarray, Affine]]
        A list of tuples, each containing the title, lake map, and transform for each panel
    model_fields: ModelFields
        The model fields object containing the model grid and other relevant data
    area: BaseGeometry
        The area of interest to plot, used to set the extent of the plots
    """
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6 * len(panels), 4.8),
        subplot_kw={"projection": EPSG3031},
        squeeze=False,
    )
    xmin, ymin, xmax, ymax = area.bounds
    for ax, (title, lake_map, transform) in zip(axes[0], panels):
        if transform is None:
            # a lake map on the model grid
            im = ax.pcolormesh(
                model_fields.lon,
                model_fields.lat,
                lake_map,
                cmap=BINARY_LAKE_CMAP,
                norm=BINARY_LAKE_NORM,
                shading="auto",
                transform=ccrs.PlateCarree(),
            )
        else:
            # satellite observation lake map
            im = ax.imshow(
                lake_map,
                extent=plotting_extent(lake_map, transform),
                origin="upper",
                cmap=BINARY_LAKE_CMAP,
                norm=BINARY_LAKE_NORM,
                transform=EPSG3031,
            )
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1], label="Lake presence")
        ax.set_extent([xmin, xmax, ymin, ymax], crs=EPSG3031)
        ax.coastlines(resolution="10m", linewidth=0.8)
        ax.gridlines(draw_labels=False, alpha=0.3)
    return fig


def _depth_hist(ax, depths, color):
    """
    Depth histogram plot
    """
    mean_depth = np.mean(depths)
    ax.hist(depths, bins=50, color=color, alpha=0.7, edgecolor="black")
    ax.axvline(mean_depth, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_depth:.2f} m")
    ax.legend(fontsize=9)


def _save_with_metadata(fig, png_path, metadata, shown=None):
    """
    Save a figure and metadata
    """
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    record_and_save(png_path=png_path, scenes=shown, **metadata)
