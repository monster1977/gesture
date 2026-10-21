import os
import sys
import traceback

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
os.chdir(_here)

import gesture_qt

try:
    gesture_qt.main()
except Exception:
    with open(os.path.join(_here, "error.log"), "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    raise
