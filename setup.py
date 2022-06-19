from setuptools import setup, Extension
import numpy as np

setup(
    name="spea2dominationscores",
    version="1.0.0",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    ext_modules=[
        Extension(
            name="spea2dominationscores",
            sources=["spea2/spea2dominationscores/spea2dominationscoresmodule.c"],
            include_dirs=[np.get_include()],
        ),
    ],
)
