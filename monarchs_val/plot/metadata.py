"""
Record some information about how a plot was generated and save it
to a JSON file alongside the image
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class DatasetInfo:
    """
    Information about a single satellite scene or model output file used in the analysis.
    """
    dataset: str  # "model", "gvi" or "antarcticlakes"
    path: str
    date: str  # YYYY-MM-DD
    offset_days: int  # how many days off the model peak this scene is


@dataclass
class Metadata:
    """
    Information about how a plot was generated.
    """

    argv: list[str]
    timestamp: str
    output_png: str
    input_paths: dict[str, str]
    parameters: dict[str, Any]
    domain: dict[str, Any]
    scenes: list[DatasetInfo] = field(default_factory=list)


def record_and_save(
    *,
    png_path,
    input_paths,
    parameters,
    domain,
    scenes=None,
):
    """
    Save a JSON file with metadata about how the plot was generated

    Parameters
    ----------
    png_path: Path
        The path to the output PNG file for which metadata is being recorded
    input_paths: dict[str, str]
        A dictionary of input file paths used in generating the plot, keyed by dataset name
    parameters: dict[str, Any]
        A dictionary of parameters used in generating the plot
    domain: dict[str, Any]
        A dictionary describing the domain of the analysis, including relevant spatial information
    scenes: list[DatasetInfo], optional
        A list of DatasetInfo objects describing the datasets used in the analysis
    """
    record = Metadata(
        argv=sys.argv[:],
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        output_png=str(png_path),
        input_paths=input_paths,
        parameters=parameters,
        domain=domain,
        scenes=scenes or [],
    )
    metadata = png_path.with_suffix(".json")
    data = asdict(record)
    metadata.write_text(json.dumps(data, indent=2, default=str))
