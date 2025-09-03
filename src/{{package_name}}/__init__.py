"""{{package_name}}: {{description}}

{{long_description}}
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("{{pypi_package_name}}")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "{{author_name}}"
__email__ = "{{author_email}}"

from .core import {{main_class}}

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "{{main_class}}",
]
