"""
Code for reading and aggregating GVI and AntarcticLakes satellite observations.

- GVI (Moussavi et al. 2020): Landsat scenes over George VI. Scenes are made up of folders with:
  - A lake mask (`*_All_Masks.tif`) (1 lake, 0 other, 2 rock or seawater, 3 cloud, -1 outside the scene)
  - Lake depths (`*_Average_Red_And_Panchromatic_Depth.tif`) in mm.

- AntarcticLakes (Dirscherl et al. 2021):
  `year_and_month_H_max_extent.tif` (1 lake, 0 not lake).

You can run this module separately to load in data from a directory - check the 
`if __name__ == "__main__":` block at the bottom for an example.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, mapping


@dataclass
class Scene:
    """
    Info about one satellite scene and its date read in from the filename
    """

    dataset: str  # "gvi" or "antarcticlakes"
    path: Path  # the lake-mask file
    date: date  # AntarcticLakes half-months are dated the 8th and the 22nd


@dataclass
class LakeMask:
    """
    Lake mask for a given scene
    """

    lake: np.ndarray  # lake mask
    # observed is either True or False for each pixel in the area of interest depending on
    # whether the pixel is a) in the area and b) lake or not 
    observed: np.ndarray  
    transform: Affine  # places the pixels in EPSG:3031


@dataclass
class AggregatedScenes:
    """
    Aggregate several scenes together and create a lake mask from them for the combined area/times
    """

    n_observed: np.ndarray 
    n_lake: np.ndarray  
    transform: Affine  # affine transform so we end up in the right CRS


def _overlaps(path, area):
    """
    Check whether a given scene overlaps an area specified by the input geometry (`area`).
    This is used to determine if a scene covers the part of George VI we are interested in.
    If no area is specified, then always True.

    Parameters
    ----------
    path: Path
        Path to the scene file
    area: BaseGeometry | None
        The area of interest to check for overlap. If None, always return True.
    """
    # maybe want to convert to EPSG:3031 rather than raise here(?)
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 3031:
            raise ValueError(f"{path}: expected a raster in EPSG:3031, got {src.crs}")
        bounds = src.bounds
    return area is None or area.intersects(box(bounds.left, bounds.bottom, bounds.right, bounds.top))


def find_gvi_scenes(gvi_dir, area):
    """
    Find all GVI lake scenes in a directory, and restrict them to a specific `area` if given.
    """
    scenes = []
    for path in Path(gvi_dir).rglob("*_All_Masks.tif"):
        # filenames are like "LC08_L1GT_213111_20131212_20170428_01_T2_All_Masks.tif".
        # the fourth part is the start date
        date_text = path.stem.split("_")[3]
        scene_date = date(int(date_text[:4]), int(date_text[4:6]), int(date_text[6:]))
        # check the scene's bounding box overlaps the area of interest.
        # if no area specified then the check always passes
        if _overlaps(path, area):
            scenes.append(Scene(dataset="gvi", path=path, date=scene_date))
    return scenes


def find_antarcticlakes_scenes(antarcticlakes_dir, area=None):
    """
    as find_gvi_scenes(), but for AntarcticLakes. Files are named `year_and_month_H_max_extent.tif`,
    with H = 1 for the first half of the month (dated the 8th) and H = 2 for the second half (dated the 22nd).

    Parameters
    ----------
    antarcticlakes_dir: str | Path
        Path to the AntarcticLakes dataset directory
    area: BaseGeometry | None
        The area of interest to check for overlap. If None, always return True.
    """
    scenes = []
    for path in Path(antarcticlakes_dir).glob("*_max_extent.tif"):
        year_and_month, which_half = path.stem.split("_")[:2]
        scene_date = date(
            int(year_and_month[:4]),
            int(year_and_month[4:]),
            8 if which_half == "1" else 22,
        )
        if _overlaps(path, area):
            scenes.append(Scene(dataset="antarcticlakes", path=path, date=scene_date))
    return scenes


def nearest_scene(scenes, target_date):
    """
    The scene nearest in date to `target_date`
    """
    offsets = [abs((scene.date - target_date).days) for scene in scenes]
    return scenes[int(np.argmin(offsets))]


def read_lake_mask(scene, area):
    """
    Read a scene's lake mask, cut to the rectangle of pixels around `area`. Pixels in that rectangle but
    outside `area` count as not observed, as do GVI cloud, rock, seawater and outside-scene pixels.
    """
    with rasterio.open(scene.path) as src:
        clipped, transform = rasterio_mask(
            src, [mapping(area)], crop=True, filled=False
        )
    inside = ~np.ma.getmaskarray(clipped[0])
    values = clipped[0].filled(0)
    observed = (
        inside & ((values == 0) | (values == 1)) if scene.dataset == "gvi" else inside
    )
    return LakeMask(lake=values == 1, observed=observed, transform=transform)


def read_gvi_depth(scene, area):
    """
    Lake depth in m for a GVI scene, restricted to `area`
    """
    depth_path = scene.path.parent / scene.path.name.replace(
        "_All_Masks.tif", "_Average_Red_And_Panchromatic_Depth.tif"
    )
    if not depth_path.exists():
        return None
    with rasterio.open(depth_path) as src:
        clipped, transform = rasterio_mask(
            src, [mapping(area)], crop=True, filled=True, nodata=-1
        )
    values = clipped[0]
    depth = values.astype(float)
    depth[values < 0] = np.nan
    return depth / 1000.0, transform


def choose_gvi_depths(scenes, area, target_date, window_days):
    """
    Select the GVI scene with the most lake pixels within `window_days` of `target_date`, and return its
    date and depths in m, restricted to `area`. None if no scene there has lake depths

    Parameters
    ----------
    scenes: list[Scene]
        The GVI scenes to choose from
    area: BaseGeometry
        The area of interest to restrict the depths to
    target_date: date
        The date to compare against
    window_days: int
        The number of days either side of `target_date` to consider
    """
    candidates = []
    for scene in scenes:
        if abs((scene.date - target_date).days) > window_days:
            continue
        gvi_depth = read_gvi_depth(scene, area)
        if gvi_depth is None:
            continue
        depth, _ = gvi_depth
        lake_depths = depth[depth > 0]
        if lake_depths.size > 0:
            candidates.append((scene.date, lake_depths))
    if not candidates:
        return None
    n_lake_pixels = [depths.size for _, depths in candidates]
    return candidates[int(np.argmax(n_lake_pixels))]


def aggregate_scenes(scenes, area, resolution):
    """
    Put several scenes of either GVI or AntarcticLakes onto one grid covering `area` with square pixels
    each `resolution` metres wide, and for each pixel count how many scenes cover that pixel and how many
    have lakes.

    Parameters
    ----------
    scenes: list[Scene]
        The scenes to aggregate
    area: BaseGeometry
        The area of interest to restrict the aggregation to
    resolution: float
        The width of each pixel in metres
    """
    xmin, ymin, xmax, ymax = area.bounds
    width, height = (
        int(np.ceil((xmax - xmin) / resolution)),
        int(np.ceil((ymax - ymin) / resolution)),
    )
    transform = from_origin(xmin, ymax, resolution, resolution)
    n_observed = np.zeros((height, width), dtype=int)
    n_lake = np.zeros((height, width), dtype=int)
    for scene in scenes:
        mask = read_lake_mask(scene, area)
        observed = np.zeros((height, width), dtype=np.uint8)
        lake = np.zeros((height, width), dtype=np.uint8)
        for scene_pixels, grid_pixels in ((mask.observed, observed), (mask.lake & mask.observed, lake)):
            reproject(
                scene_pixels.astype(np.uint8),
                grid_pixels,
                src_transform=mask.transform,
                src_crs="EPSG:3031",
                dst_transform=transform,
                dst_crs="EPSG:3031",
                resampling=Resampling.nearest,
            )
        n_observed += observed
        n_lake += lake
    return AggregatedScenes(n_observed=n_observed, n_lake=n_lake, transform=transform)



if __name__ == "__main__":
    # Example: where the GVI scenes from a date range show lakes on George VI Ice Shelf. Each scene is read at
    # its full 30 m resolution, so a month of scenes takes about 30 s and a few GB of memory.
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from pyproj import Transformer
    from shapely.geometry import Polygon

    gvi_dir = "validation/GVI"
    start_date = date(2020, 1, 1)
    end_date = date(2020, 1, 31)

    # area of interest around GVI ice shelf. you could also just use `shapely.geometry.box` if you have a 
    # rectangular domain of interest, but here we are using a funky CRS so a rectangle doesnt quite work 
    # for the actual validation scripts I read the bounding box directly from the model grid, which is in turn
    # determined via the input DEM
    to_3031 = Transformer.from_crs(4326, 3031, always_xy=True)
    xs, ys = to_3031.transform([-67.165, -66.034, -68.751, -69.778], [-71.189, -72.014, -72.361, -71.519])
    area = Polygon(zip(xs, ys))

    # put scenes in the date range onto a grid and count the lakes
    scenes = [s for s in find_gvi_scenes(gvi_dir, area) if start_date <= s.date <= end_date]
    counts = aggregate_scenes(scenes, area, resolution=100)
    print(f"{len(scenes)} GVI scenes. Lake seen at least once over {(counts.n_lake > 0).sum() / 100:.0f} km2")

    # lakes in blue, land light grey
    polar_stereo = ccrs.SouthPolarStereo(true_scale_latitude=-71)  # EPSG:3031
    xmin, ymin, xmax, ymax = area.bounds
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection=polar_stereo)
    ax.set_extent((xmin, xmax, ymin, ymax), crs=polar_stereo)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="lightgrey")
    ax.coastlines(resolution="10m", linewidth=0.6)
    # find lakes and plot
    lake = np.where(counts.n_lake > 0, 1.0, np.nan) 
    ax.imshow(
        lake, extent=(xmin, xmax, ymin, ymax), transform=polar_stereo, cmap=ListedColormap(["#2166ac"]),
        interpolation="none", zorder=2,  # zorder 2 draws the lakes on top of the land
    )
    ax.set_title(f"GVI lakes (blue), {start_date} to {end_date}")
    # lat/lon gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.3)
    # ticks
    gl.xlocator = plt.FixedLocator([-69, -68, -67])
    gl.ylocator = plt.FixedLocator([-72, -71, -70])
    plt.tight_layout() 
    plt.show()
