"""
Common sys.path setup for all RAD_APP entry-point scripts.

Usage (at the top of any scripts/*.py):
    from modelscope._path_setup import setup_sys_path
    setup_sys_path()

This ensures RAD_APP/, RAD/ (root), and RAD_reproduction/ are on sys.path
so that `import modelscope` and `import dinov3` both work.
"""

import os
import sys


_CALLED = False


def setup_sys_path():
    global _CALLED
    if _CALLED:
        return
    _CALLED = True

    _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _root_dir = os.path.dirname(_project_dir)

    for p in [_project_dir, _root_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    _data_dir = os.path.join(_root_dir, 'RAD_reproduction')
    if _data_dir not in sys.path:
        sys.path.append(_data_dir)
