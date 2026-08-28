# Copyright (c) OpenMMLab. All rights reserved.
import mmcv

from .version import __version__, short_version

import sys
import types

# handle retrocompatibility MMCV 2.x / MMEngine -> MMCV 1.x
try:
    import mmcv.runner
except ImportError:
    try:
        import mmengine.model as mmengine_model
        import mmengine.runner as mmengine_runner

        # dummy virtual module called 'mmcv.runner'
        compat_runner = types.ModuleType('mmcv.runner')

        # Copy attributes from both mmengine.model and mmengine.runner
        compat_runner.__dict__.update(mmengine_model.__dict__)
        compat_runner.__dict__.update(mmengine_runner.__dict__)

        # Inject the unified module
        sys.modules['mmcv.runner'] = compat_runner

        # 'mmcv.runner' is accessible directly from the mmcv object
        if 'mmcv' in sys.modules:
            sys.modules['mmcv'].runner = compat_runner

    except ImportError as e:
        raise ImportError(
            "Cannot find either 'mmcv.runner' (MMCV 1.x) or 'mmengine' (MMCV 2.x). "
            "Make sure MMEngine is installed in the environment."
        ) from e

def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.3.17'
mmcv_maximum_version = '2.3.0' # updated for blackwell 12.0
mmcv_version = digit_version(mmcv.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

__all__ = ['__version__', 'short_version']
