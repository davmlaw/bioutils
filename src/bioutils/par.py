"""Pseudoautosomal region (PAR) coordinates for human reference assemblies.

The pseudoautosomal regions are the segments at either end of X and Y that
recombine and are diploid in both sexes. Code that treats X or Y as hemizygous
in males, such as sex inference, copy-number estimation or variant calling
with per-sample ploidy, has to mask them first.

Coordinates are **interbase** (0-based, left-closed, right-open), the same
convention as :mod:`bioutils.cytobands` and the ``_i`` system described in
:mod:`bioutils.coordinates`. NCBI publishes the regions 1-based inclusive; the
generator converts ``[start, end]`` to ``[start - 1, end]``, so a
1-based position ``pos`` (for example a VCF ``POS``) is inside a region
``[s, e]`` when ``s <= pos - 1 < e``. The original NCBI values are recoverable
as ``[start + 1, end]`` and are recorded in ``tests/test_par.py``.

Chromosome names are the assembly's own (``"X"``, ``"Y"``), without a ``chr``
prefix, as elsewhere in bioutils (see the note in :mod:`bioutils.assemblies`).

Region labels are ``"PAR1"`` (the p-terminal region, shared by the short arms)
and ``"PAR2"`` (the q-terminal region): NCBI's own "PAR#1" and "PAR#2" names
without the ``#``, which gives the plain forms used in the literature and by
other resources.

Data is generated from each assembly's ``assembly_regions.txt`` report,
published by NCBI beside the assembly report, whose ``PAR`` rows give the
regions::

  ./sbin/ncbi-par-to-json GRCh38.p14 GCF_000001405.40_GRCh38.p14_assembly_regions.txt \\
      | gzip -c >src/bioutils/_data/par/GRCh38.json.gz

PAR coordinates are stable across patch releases within a build, so files are
named for the build (``"GRCh38"``). Functions take an assembly name as used by
:func:`bioutils.assemblies.get_assembly`, either a build or a patch release
(``"GRCh38.p14"``), which resolves to its build. Each file is a flat JSON document in the style of
``_data/assemblies``: provenance fields (``assembly``, ``refseq_ac``,
``accessions``, ``coordinates``, ``source``, ``source_url``) beside the
``regions`` data (see :func:`get_par_meta`).
"""

import functools
import gzip
import json
import re
from importlib import resources
from pathlib import Path
from typing import TypedDict, cast

_data_dir = Path(str(resources.files("bioutils") / "_data" / "par"))


class ParMeta(TypedDict):
    """Provenance fields of a PAR dataset; see :func:`get_par_meta`."""

    assembly: str
    refseq_ac: str
    accessions: dict[str, str]
    coordinates: str
    source: str
    source_url: str


def _build_name(assembly_name: str) -> str:
    """Maps an assembly name to the build its PAR data is stored under."""
    build = re.sub(r"\.p\d+$", "", assembly_name)
    if build not in get_par_names():
        msg = f"no PAR data for assembly {assembly_name!r}; available builds: {sorted(get_par_names())}"
        raise ValueError(msg)
    return build


@functools.cache
def _load(assembly_name: str) -> dict:
    fn = _data_dir / f"{_build_name(assembly_name)}.json.gz"
    with gzip.open(fn, mode="rt", encoding="utf-8") as fh:
        return json.load(fh)


def get_par_names() -> list[str]:
    """Retrieves available PAR datasets from the ``_data/par`` directory.

    Returns:
        list of str: The builds with PAR data. Any patch release of these
        (e.g. ``"GRCh38.p14"``) is also accepted by the other functions.

    Examples:
        >>> sorted(get_par_names())
        ['GRCh37', 'GRCh38', 'T2T-CHM13v2.0']
    """

    return [n.name.replace(".json.gz", "") for n in _data_dir.glob("*.json.gz")]


def get_par_map(assembly_name: str) -> dict[str, dict[str, list[int]]]:
    """Retrieves PAR coordinates for an assembly.

    Args:
        assembly_name (str): A build (``"GRCh38"``) or patch release
            (``"GRCh38.p14"``); patches share their build's coordinates.

    Returns:
        dict: A dictionary of the form ``{chromosome: {label: [start_i, end_i]}}``,
        with interbase (0-based, half-open) coordinates.

    Examples:
        >>> par = get_par_map("GRCh38")
        >>> par["X"]["PAR1"]
        [10000, 2781479]
        >>> par["Y"]["PAR2"]
        [56887902, 57217415]

        A patch release gives its build's coordinates:

        >>> get_par_map("GRCh38.p14") == par
        True

        PAR1 is not at the same X and Y coordinates in every build:

        >>> get_par_map("GRCh37")["X"]["PAR1"]
        [60000, 2699520]
        >>> get_par_map("GRCh37")["Y"]["PAR1"]
        [10000, 2649520]
    """

    return _load(assembly_name)["regions"]


def get_par_meta(assembly_name: str) -> ParMeta:
    """Retrieves provenance metadata for a PAR dataset.

    Args:
        assembly_name (str): A build or patch release, as for :func:`get_par_map`.

    Returns:
        dict: Every top-level field of the dataset except ``regions``: the
        assembly the data was generated from and its RefSeq accession, the X and Y sequence accessions,
        the coordinate convention, and the upstream source and URL.

    Examples:
        >>> meta = get_par_meta("GRCh38")
        >>> meta["assembly"]
        'GRCh38.p14'
        >>> meta["refseq_ac"]
        'GCF_000001405.40'
        >>> meta["accessions"]["X"]
        'NC_000023.11'
        >>> meta["coordinates"]
        'interbase (0-based, left-closed, right-open)'
    """

    return cast("ParMeta", {k: v for k, v in _load(assembly_name).items() if k != "regions"})


def get_par_maps(
    assembly_names: list[str] | None = None,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Retrieves PAR coordinates for multiple datasets.

    If datasets are not specified, retrieves data for all available ones.

    Args:
        assembly_names (list of str, optional): Builds or patch releases, as
            for :func:`get_par_map`.

    Returns:
        dict: A dictionary of the form ``{assembly_name: par_data}``.

    Examples:
        >>> maps = get_par_maps()
        >>> maps["GRCh38"]["X"]["PAR1"]
        [10000, 2781479]
    """

    if assembly_names is None:
        assembly_names = get_par_names()
    return {name: get_par_map(name) for name in assembly_names}


def get_par_label(assembly_name: str, chrom: str, pos_i: int) -> str | None:
    """Names the pseudoautosomal region containing a position, if any.

    Args:
        assembly_name (str): A build or patch release, as for :func:`get_par_map`.
        chrom (str): Chromosome name as used by the assembly (``"X"`` or
            ``"Y"``; no ``chr`` prefix).
        pos_i (int): Interbase (0-based) position of a nucleotide, i.e. the
            interval ``[pos_i, pos_i + 1)``. For a 1-based position such as a
            VCF ``POS``, pass ``POS - 1``.

    Returns:
        str or None: ``"PAR1"`` or ``"PAR2"`` when the position falls inside a
        region, otherwise None. None for any chromosome that has no
        pseudoautosomal regions.

    Raises:
        ValueError: If ``chrom`` carries a ``chr`` prefix, or there is no PAR
            data for ``assembly_name``.

    Examples:
        >>> get_par_label("GRCh38", "X", 1000000)
        'PAR1'
        >>> get_par_label("GRCh38", "X", 155800000)
        'PAR2'

        The non-PAR bulk of X, where males are hemizygous:

        >>> get_par_label("GRCh38", "X", 100000000) is None
        True

        Autosomes never are:

        >>> get_par_label("GRCh38", "1", 1000000) is None
        True
    """

    if chrom.lower().startswith("chr"):
        msg = f"use the assembly's chromosome name without a 'chr' prefix, got {chrom!r}"
        raise ValueError(msg)
    regions = get_par_map(assembly_name).get(chrom, {})
    for label, (start_i, end_i) in regions.items():
        if start_i <= pos_i < end_i:
            return label
    return None


def in_par(assembly_name: str, chrom: str, pos_i: int) -> bool:
    """Tests whether a position falls in a pseudoautosomal region.

    Args:
        assembly_name (str): A build or patch release, as for :func:`get_par_map`.
        chrom (str): Chromosome name as used by the assembly (``"X"`` or
            ``"Y"``; no ``chr`` prefix).
        pos_i (int): Interbase (0-based) position of a nucleotide. For a
            1-based position such as a VCF ``POS``, pass ``POS - 1``.

    Returns:
        bool: True if the position is within a PAR on ``chrom``. Always False
        for chromosomes that have no pseudoautosomal regions.

    Raises:
        ValueError: If ``chrom`` carries a ``chr`` prefix, or there is no PAR
            data for ``assembly_name``.

    Examples:
        The first and last bases of GRCh38 PAR1 are NCBI 10,001 and 2,781,479
        (1-based), so in interbase terms:

        >>> in_par("GRCh38", "X", 10000)
        True
        >>> in_par("GRCh38", "X", 9999)
        False
        >>> in_par("GRCh38", "X", 2781478)
        True
        >>> in_par("GRCh38", "X", 2781479)
        False

        From a VCF record:

        >>> vcf_pos = 10001
        >>> in_par("GRCh38", "X", vcf_pos - 1)
        True
    """

    return get_par_label(assembly_name, chrom, pos_i) is not None
