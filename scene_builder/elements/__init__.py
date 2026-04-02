"""Scene element registry and built-in definitions."""
from .registry import get_scene_element, list_scene_elements, iter_scene_elements
from . import shape as _shape  # noqa: F401
from . import light as _light  # noqa: F401
from . import builtin as _builtin  # noqa: F401
from .shape import *  # noqa: F401,F403
from .light import *  # noqa: F401,F403
from .builtin import *  # noqa: F401,F403

__all__ = [
    'get_scene_element',
    'list_scene_elements',
    'iter_scene_elements',
] + _shape.__all__ + _light.__all__ + _builtin.__all__
