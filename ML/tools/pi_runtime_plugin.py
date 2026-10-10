"""pytest -p tools.pi_runtime_plugin: enforce classical Pi runtime imports.

Training tests using importorskip are skipped; runtime tests are still run.
This tests dependencies, not Raspberry Pi speed or hardware compatibility.
"""
import importlib.abc
import sys

BLOCKED = {'sklearn','torch','onnx','onnxruntime','scipy','matplotlib','pandas'}

class PiImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise ModuleNotFoundError('Pi simulation blocked ' + fullname, name=fullname)

def pytest_configure(config):
    loaded = {name.split('.')[0] for name in sys.modules} & BLOCKED
    if loaded:
        raise RuntimeError('forbidden packages loaded before Pi guard: ' + str(loaded))
    sys.meta_path.insert(0,PiImports())
