from setuptools import setup, find_packages

setup(
    name="yolov5_simple",
    version="0.0.1",
    description='Simple implementation of YOLOv5',
    url="https://github.com/BlakeJC94/yolov5-simple",
    author="BlakeJC94",
    classifiers=[
        'Development Status :: Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        'clearml',
        'fire',
        'humanize',
        'numpy',
        'pandas',
        'pytorch-lightning',
        'scikit-learn',
        'scipy',
        'torch',
        'torchvision',
        'torchmetrics',
        'tqdm',
    ],
    # entry_points={
    #     'console_scripts': ['yolov5=yolov5_simple.__main__:main'],
    # },
)

