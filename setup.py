'''
Author: yingmuzhi cyxscj@126.com
Date: 2026-03-09 15:32:01
LastEditors: yingmuzhi cyxscj@126.com
LastEditTime: 2026-03-11 09:39:59
FilePath: /CLIEM_code/setup.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_readme() -> str:
    readme = ROOT / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="cliem-code",
    version="0.1.0",
    description="SEM preprocess scripts for CLIEM_code",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yingmuzhi/CLIEM",
    packages=find_packages(include=["Preprocess", "Preprocess.*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "tifffile",
        "Pillow",
        "scipy",
        "pandas",
        "tqdm",
        "natsort",
        "matplotlib",
        "seaborn",
        "ipykernel",
        "imagecodecs",
        "xlrd",
    ],
    entry_points={
        "console_scripts": [
            "cliem-pipeline=Preprocess.pipeline:main",
        ]
    },
)

