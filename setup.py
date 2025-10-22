"""
PlantHealth AI - Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Ler o README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Ler requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='planthealth-ai',
    version='1.0.0',
    author='Ricardo Frugoni',
    author_email='ricardo@codex.ai',
    description='Sistema Inteligente de Detecção de Doenças em Plantas usando Deep Learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ricardofrugoni/planthealth-ai',
    project_urls={
        'Bug Reports': 'https://github.com/ricardofrugoni/planthealth-ai/issues',
        'Source': 'https://github.com/ricardofrugoni/planthealth-ai',
        'Documentation': 'https://github.com/ricardofrugoni/planthealth-ai#readme',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'flake8>=6.0.0',
            'black>=23.0.0',
            'mypy>=1.0.0',
        ],
        'gpu': [
            'tensorflow-gpu>=2.13.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'planthealth-gradio=app.gradio_app:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='deep-learning computer-vision plant-disease agriculture transfer-learning tensorflow keras',
)

