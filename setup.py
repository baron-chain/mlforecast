import setuptools
from configparser import ConfigParser
from pkg_resources import parse_version

# Ensure setuptools version is at least 36.2
assert parse_version(setuptools.__version__) >= parse_version('36.2'), "setuptools>=36.2 is required"

# Read configuration from settings.ini
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

# Validate expected settings
cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for key in expected:
    assert key in cfg, f"Missing expected setting: {key}"

# Setup configuration
setup_cfg = {key: cfg[key] for key in cfg_keys}

# License configuration
licenses = {
    'apache2': ('Apache Software License 2.0', 'OSI Approved :: Apache Software License'),
    'mit': ('MIT License', 'OSI Approved :: MIT License'),
    'gpl2': ('GNU General Public License v2', 'OSI Approved :: GNU General Public License v2 (GPLv2)'),
    'gpl3': ('GNU General Public License v3', 'OSI Approved :: GNU General Public License v3 (GPLv3)'),
    'bsd3': ('BSD License', 'OSI Approved :: BSD License'),
}

# Development status
statuses = [
    '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive'
]

# Python versions
py_versions = '3.8 3.9 3.10 3.11'.split()

# Requirements
requirements = cfg['requirements'].split()
extra_requirements = {
    'dask': cfg['dask_requirements'].split(),
    'ray': cfg['ray_requirements'].split(),
    'spark': cfg['spark_requirements'].split(),
    'aws': cfg['aws_requirements'].split(),
    'azure': cfg['azure_requirements'].split(),
    'gcp': cfg['gcp_requirements'].split(),
    'polars': cfg['polars_requirements'].split(),
    'dev': cfg['dev_requirements'].split(),
    'lag_transforms': [],
}
extra_requirements['all'] = set().union(*extra_requirements.values())

# License
min_python = cfg['min_python']
license_info = licenses.get(cfg['license'].lower(), (cfg['license'], None))

# Setup function
setuptools.setup(
    name='mlforecast',
    license=license_info[0],
    classifiers=[
        f'Development Status :: {statuses[int(cfg["status"])]}',
        f'Intended Audience :: {cfg["audience"].title()}',
        f'Natural Language :: {cfg["language"].title()}',
        *[f'Programming Language :: Python :: {v}' for v in py_versions[py_versions.index(min_python):]],
        *(f'License :: {license_info[1]}',) if license_info[1] else ()
    ],
    url=cfg['git_url'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require=extra_requirements,
    dependency_links=cfg.get('dep_links', '').split(),
    python_requires=f'>={cfg["min_python"]}',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    zip_safe=False,
    entry_points={
        'console_scripts': cfg.get('console_scripts', '').split(),
        'nbdev': [f'{cfg.get("lib_path")}={cfg.get("lib_path")}._modidx:d']
    },
    **setup_cfg
)
