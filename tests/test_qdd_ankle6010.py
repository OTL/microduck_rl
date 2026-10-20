"""Ankle option D (GIM6010-8, robo PR #66): QDD_ANKLE=GIM6010-8 switches the model."""

import json
import os
import subprocess
import sys

CODE = """
import json, math
from mjlab_microduck.robot import qdd_biped_constants as C
print(json.dumps(dict(xml=C.QDD_BIPED_XML.name, exists=C.QDD_BIPED_XML.exists(), rated=C.ANKLE_RATED_NM,
    suffix=C.EXPERIMENT_SUFFIX, stand=C.STAND_BASE_Z, knee=math.degrees(C.knee_max(math.radians(-50))))))
"""


def _consts(ankle: str) -> dict:
    env = dict(os.environ, QDD_ANKLE=ankle)
    out = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_ankle_switch():
    a, d = _consts("GIM3510-8"), _consts("GIM6010-8")
    assert a["xml"] == "qdd_biped.xml" and a["rated"] == 1.0 and a["suffix"] == ""
    assert d["xml"] == "qdd_biped_ankle6010.xml" and d["exists"] and d["rated"] == 5.0 and d["suffix"] == "_ankle6010"
    assert abs(d["stand"] - a["stand"] - 0.012) < 1e-6  # foot hangs 12 mm lower
    assert abs(a["knee"] - d["knee"] - 10.0) < 1e-6  # knee folds 10 deg less
