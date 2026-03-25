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

try:
    from ._version import __version__
except ImportError:
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
