# -*- coding: utf-8 -*-

from os.path import dirname, join
from setuptools import find_packages, setup

from tfos import VERSION

__version__ = VERSION
__version_str__ = '.'.join(map(str, VERSION))

PATH = dirname(__file__)
f = open(join(PATH, 'README.md'))
long_description = f.read().strip()
f.close()

install_requires = [
    'tensorflow',
    'tensorflowonspark'
]
# dependency_links = []
# tests_require = []

setup(
    name="tfos",
    description="tensorflow on spark",
    license="",
    url="https://gitlab.zzjz.com/tfos/tfos",
    # long_description=long_description,
    version=__version_str__,
    author="weijinlong",
    author_email="jinlong.wei@zzjunzhi.com",
    packages=find_packages(
        where='.',
        # exclude=('test', 'examples'),
        include=['tfos']
    ),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
    install_requires=install_requires,
    include_package_data=True,  # #启用清单文件MANIFEST.in
    # test_suite='nose.collector',
    entry_points={},
)
