"""
Static assets served inside the image summary report.

Kept as a regular package -- rather than a bare directory -- so that
``importlib.resources.files("pytest_pyvista.summary.assets")`` resolves through a
plain package loader on every supported Python, instead of relying on namespace
package resource readers.
"""

from __future__ import annotations
