import multiprocessing as mp
import platform
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from numpy import get_include as np_get_include


class SetupSettings:

    def __init__(self):
        self.os = platform.system().lower()

    @property
    def nthreads(self) -> int:
        return {
            "linux": mp.cpu_count(),
            "darwin": mp.cpu_count(),
            "windows": 0
        }.get(self.os, 0)


# fmt: off
cython_modules = [
    ("causalml.inference.tree._tree._tree", "causalml/inference/tree/_tree/_tree.pyx"),
    ("causalml.inference.tree._tree._criterion", "causalml/inference/tree/_tree/_criterion.pyx"),
    ("causalml.inference.tree._tree._splitter", "causalml/inference/tree/_tree/_splitter.pyx"),
    ("causalml.inference.tree._tree._utils", "causalml/inference/tree/_tree/_utils.pyx"),
    ("causalml.inference.tree.causal._criterion", "causalml/inference/tree/causal/_criterion.pyx"),
    ("causalml.inference.tree.causal._builder", "causalml/inference/tree/causal/_builder.pyx"),
    ("causalml.inference.tree.uplift", "causalml/inference/tree/uplift.pyx"),
]
# fmt: on


settings = SetupSettings()

extensions = [
    Extension(
        name,
        [source],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    )
    for name, source in cython_modules
]

packages = find_packages(exclude=["tests", "tests.*"])

setup(
    packages=packages,
    ext_modules=cythonize(extensions, annotate=True, nthreads=settings.nthreads),
    include_dirs=[np_get_include()],
)
