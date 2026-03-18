from setuptools import setup, find_packages

setup(
    name="mesh_parameterization",
    version="0.1.0",
    description=(
        "Quasi-isometric mesh parameterization using heat-based geodesics "
        "and classical MDS (based on MDPI Mathematics 7(8):753, 2019)"
    ),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "trimesh[easy]>=3.15.0",
    ],
)
