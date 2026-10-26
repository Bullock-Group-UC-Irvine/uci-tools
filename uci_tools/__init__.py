from . import (
    tools,
    staudt_tools,
    fire_io,
    rotate_galaxy,
    vel_map,
    fire_tools,
    firebox_io,
    config,
)

import importlib.metadata
try:
    __version__ = importlib.metadata.version('uci_tools')
except importlib.metadata.PackageNotFoundError:
    __version__ = 'unknown'

# List the modules and objects you want to make available when using wildcard 
# imports
__all__ = [
    'tools',
    'staudt_tools',
    'fire_io',
    'rotate_galaxy',
    'vel_map',
    'fire_tools',
    'firebox_io',
]
