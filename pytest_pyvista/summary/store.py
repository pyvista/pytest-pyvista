"""On-disk image store for the summary report."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Literal

from PIL import Image

FullSizeMode = Literal["none", "failing", "all"]

_IMAGES_SUBDIR = "images"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
# ``~`` is unsafe by the pattern above, so it can never appear in an unaltered name's
# slug -- which makes a disambiguated slug structurally unable to collide with a plain one.
_DIGEST_SEPARATOR = "~"
_DIGEST_LENGTH = 12


def slugify(name: str) -> str:
    """
    Reduce a test or image name to something safe to use as a filename.

    The substitution is lossy -- ``"a/b"`` and ``"a_b"`` both reduce to ``"a_b"`` -- so a
    short digest of the original name is appended whenever the name had to be altered,
    keeping two different names from claiming one filename. Names that are already safe
    are returned unchanged so the common case stays readable. The digest depends only on
    the name, so xdist workers sharing one image directory all agree on it.
    """
    slug = _UNSAFE.sub("_", name).strip("_")
    if slug == name:
        return slug
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{slug}{_DIGEST_SEPARATOR}{digest}"


class ReportImageStore:
    """Writes the report's copies of baseline, generated and difference images."""

    def __init__(self, report_dir: Path, *, max_image_size: int = 400, full_size: FullSizeMode = "failing") -> None:
        """
        Initialize the ReportImageStore.

        Parameters
        ----------
        report_dir
            Directory where images will be stored.
        max_image_size
            Maximum edge size for thumbnail images (default 400).
        full_size
            Full-size retention mode: "none", "failing", or "all" (default "failing").

        """
        self.report_dir = Path(report_dir)
        self.max_image_size = max_image_size
        self.full_size = full_size
        self.images_dir = self.report_dir / _IMAGES_SUBDIR

    def save_file(self, source: Path, slug: str, role: str, *, status: str) -> tuple[str, str | None]:
        """
        Store an image read from ``source``.

        Parameters
        ----------
        source
            Path to the source image file.
        slug
            Image slug/identifier (will be slugified).
        role
            Role of the image (e.g., "baseline", "generated", "diff").
        status
            Status of the test (e.g., "passed", "failed").

        Returns
        -------
        tuple[str, str | None]
            A tuple of (thumbnail_relpath, full_relpath_or_None) as POSIX paths
            relative to the report directory.

        """
        with Image.open(source) as image:
            return self.save_image(image, slug, role, status=status)

    def save_image(self, image: Image.Image, slug: str, role: str, *, status: str) -> tuple[str, str | None]:
        """
        Store a downscaled copy of ``image``, plus a full-resolution copy when retained.

        Parameters
        ----------
        image
            PIL Image object to store.
        slug
            Image slug/identifier (will be slugified).
        role
            Role of the image (e.g., "baseline", "generated", "diff").
        status
            Status of the test (e.g., "passed", "failed").

        Returns
        -------
        tuple[str, str | None]
            A tuple of (thumbnail_relpath, full_relpath_or_None) as POSIX paths
            relative to the report directory. ``full_relpath_or_None`` is ``None``
            when this record does not retain a full-resolution copy.

        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        safe = slugify(slug)
        image = image.convert("RGB")

        full_relpath: str | None = None
        if self._keeps_full_size(status):
            full_name = f"{safe}.{role}.full.png"
            image.save(self.images_dir / full_name)
            full_relpath = f"{_IMAGES_SUBDIR}/{full_name}"

        thumbnail = image.copy()
        thumbnail.thumbnail((self.max_image_size, self.max_image_size))
        thumb_name = f"{safe}.{role}.png"
        thumbnail.save(self.images_dir / thumb_name)

        return f"{_IMAGES_SUBDIR}/{thumb_name}", full_relpath

    def _keeps_full_size(self, status: str) -> bool:
        """
        Determine whether to keep a full-size copy based on full_size mode.

        Parameters
        ----------
        status
            Status of the test.

        Returns
        -------
        bool
            True if a full-size copy should be retained.

        """
        if self.full_size == "all":
            return True
        if self.full_size == "none":
            return False
        return status != "passed"
