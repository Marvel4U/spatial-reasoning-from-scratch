"""worldsnap — reproducible *world snapshots* of real city districts.

Each snapshot is a set of vector layers clipped to one district, stored in a
projected CRS, accompanied by a manifest recording provenance (source, licence,
vintage, checksum) so that every downstream number in the project can be traced
back to a specific download of a specific dataset version.

Stage 1 of the pipeline: buildings (3DBAG) and street surfaces (BGT).
"""

__all__ = ["config", "manifest", "render"]
