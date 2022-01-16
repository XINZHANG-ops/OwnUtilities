import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='xin_util',
    version='1.4.26',
    author="Xin Zhang",
    author_email="1528371521zx@gmail.com",
    description="Xin's self created helper functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/XINZHANG-ops/OwnUtilities",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[
        'xin_util', 'model_trainingtime_prediction', 'hyperparam_tuning', 'data_process',
        'nlp_utils', 'raise_error', 'time_series', 'auto_widgets', 'aws_gdrive', 'ModelHub',
        'object_detect'
    ],
    license='MIT',
    # install_requires=['boto3',
    #                     'fasttext',
    #                     'gensim',
    #                     'matplotlib',
    #                     'nltk',
    #                     'numpy',
    #                     'pandas',
    #                     'scikit-learn',
    #                     'scipy',
    #                     'spacy',
    #                     'seaborn',
    #                     'sklearn',
    #                     'tqdm',
    #                   'keras'],
    include_package_data=True,
    py_modules=['utils', 'timeseries', 'textclassification'],
    package_data={'': ['data/*.json']}
)
