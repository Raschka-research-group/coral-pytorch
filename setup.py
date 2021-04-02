# Sebastian Raschka 2020
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import coral_pytorch

VERSION = coral_pytorch.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')


setup(name='coral_pytorch',
      version=VERSION,
      description='CORAL ordinal regression for PyTorch',
      author='Sebastian Raschka',
      author_email='mail@sebastianraschka.com',
      url='https://github.com/raschka-research-group/coral_pytorch',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      setup_requires=[],
      install_requires=install_reqs,
      extras_require={'testing': ['pytest'],
                      'docs': ['mkdocs']},
      license='MIT',
      platforms='any',
      keywords=['deep learning', 'pytorch', 'AI'],
      classifiers=[
             'License :: OSI Approved :: MIT License',
             'Development Status :: 5 - Production/Stable',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS',
             'Programming Language :: Python :: 3.7',
             'Topic :: Scientific/Engineering',
             'Topic :: Scientific/Engineering :: Artificial Intelligence',
             'Topic :: Scientific/Engineering :: Information Analysis',
             'Topic :: Scientific/Engineering :: Image Recognition',
      ],
      long_description_content_type='text/markdown',
      long_description="""

Library implementing the core utilities for the
CORAL ordinal regression approach from

Wenzhi Cao, Vahid Mirjalili, Sebastian Raschka (2020):
Rank Consistent Ordinal Regression for Neural Networks
with Application to Age Estimation.
Pattern Recognition Letters.
https://doi.org/10.1016/j.patrec.2020.11.008.
""")
