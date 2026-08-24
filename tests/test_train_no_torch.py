import subprocess
import sys


def test_torch_free_layers_do_not_import_torch():
    """The config/fingerprint/cost/selection/report/predict layers must import without torch, so
    the pre-flight gate (and CI) never need the `train` extra."""
    code = (
        "import sys;"
        "import polarity.train.config, polarity.train.cost, polarity.train.selection,"
        " polarity.train.report, polarity.train.predict, polarity.calibration.fit;"
        "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m);"
        "assert 'transformers' not in sys.modules;"
        "print('torch-free OK')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "torch-free OK" in out.stdout
