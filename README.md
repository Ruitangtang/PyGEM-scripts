# FA_AdaPBS_RT — Frontal Ablation Calibration and Simulation Scripts

[![Paper](https://img.shields.io/badge/📄-GMD_Preprint_2026-blue)](https://doi.org/10.5194/egusphere-2026-1081)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18761729-blue.svg)](https://doi.org/10.5281/zenodo.18761729)
[![GitHub release](https://img.shields.io/github/v/release/Ruitangtang/PyGEM-scripts?label=stable&color=blue)](https://github.com/Ruitangtang/PyGEM-scripts/releases/tag/v1.0.0-zenodo-fa-adapbs-rt)

This branch contains the **calibration and simulation scripts** used for the frontal ablation study:

> *"Joint Bayesian Calibration of Frontal Ablation and Surface Mass Balance in Global Glacier Models"* (GMD Preprint, 2026)

**Key scripts:**
- `run_calibration_AMIS_SERMeQ.py` — Bayesian calibration (AdaPBS) of frontal ablation parameters (2000–2010)
- `run_simulation_AMIS_SERMeQ.py` — Future projections (2000–2100) under CMIP6 scenarios (12 GCMs, SSP126/SSP585)
- Bash scripts for parallel regional and single-glacier runs
- Jupyter notebooks to reproduce all paper figures (Fig. 1–5)

**Full research archive:** [**Zenodo research package — concept DOI 10.5281/zenodo.18761729**](https://doi.org/10.5281/zenodo.18761729)

**Stable release:** [`v1.0.0-zenodo-fa-adapbs-rt`](https://github.com/Ruitangtang/PyGEM-scripts/releases/tag/v1.0.0-zenodo-fa-adapbs-rt)

🚀 Lightweight showcase repo: [**Ruitangtang/frontal-ablation-glacier-demo**](https://github.com/Ruitangtang/frontal-ablation-glacier-demo) — YAML configs, dry-run quickstart commands, tests, method/results visuals, and reproducibility links.

**Citation:**
> [Authors]. *Joint Bayesian Calibration of Frontal Ablation and Surface Mass Balance in Global Glacier Models*. GMD Preprint, 2026. DOI: [`10.5194/egusphere-2026-1081`](https://doi.org/10.5194/egusphere-2026-1081)

**Contact:**
> For questions, bug reports, or collaboration inquiries, please open a GitHub issue in this repository 




---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
<!--HERE GOES THE DESCRIPTION ON HOW TO SETUP CODE AND DATA FOR THE PAPER.-->
# [Frontalablation_Modeling_SERMeQ_PyGEM_OGGM] v[1.0]

## Description
This open-source glacier evolution model, written in Python, couples the calving model SERMeQ with the glacier mass-balance models PyGEM and OGGM. Model calibration is performed using an Adapted Particle Batch Smoother. The model timestep is set as monthly.

## Corresponding author
- Ruitang Yang (ORCID: [0000-0001-5145-940X])

## Requirements and Installation

### Requirements
- THe model was fully tested on Linux system
- same to [OGGM](https://docs.oggm.org/en/latest/installing-oggm.html), and the recommended environment file (*.yml) as follows, installed by [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://anaconda.org/channels/conda-forge/packages/mamba/overview) : (conda/mamba env create -f oggm_env.yml): 
```yaml
name: oggm_env
channels:
  - conda-forge
dependencies:
  - python = 3.11
  - numpy
  - scipy
  - pandas
  - shapely
  - matplotlib
  - Pillow
  - netcdf4
  - scikit-image
  - configobj
  - xarray
  - pytest
  - dask
  - bottleneck
  - pyproj
  - cartopy
  - geopandas
  - rasterio
  - rioxarray
  - seaborn
  - pytables
  - salem
  - motionless
  - h5py
  - pip
  - pip:
    - joblib
    - progressbar2
```

### Installation
- The installation is adapted from the procedures used by [PyGEM](https://pygem.readthedocs.io/en/latest/install_pygem.html) and [OGGM](https://docs.oggm.org/en/latest/installing-oggm.html), and is currently provided as a development installation. It has not yet been integrated into the stable releases of OGGM and PyGEM, but we hope to achieve this in the near future. You need [git](https://git-scm.com/) software.

#### 1. OGGM Install (dev+get access to the OGGM code)

##### activate the env
```
conda activate oggm_env
```
##### clone the repo
```
git clone https://github.com/Ruitangtang/oggm.git
```
##### get the update
```
cd oggm
git fetch origin
```
##### install the oggm (dev)
```
pip install -e .
```
##### verification (the oggm version should be the version 1.6+dev)
```
cd /tmp
python -c "
import oggm
print('=== FINAL RESULT ===')
print(f'Version: {oggm.__version__}')
print(f'Source: {oggm.__file__}')
"
```
##### Test oggm
```
pytest.oggm  --disable-warnings

```
##### load and checkout to the frontal ablation branch
```
git fetch --all
git checkout SERMeQ_RT
```
#### 2. PyGEM and PyGEM-script install
##### PyGEM
```
git clone git@github.com:Ruitangtang/PyGEM.git PyGEM_All
git fetch -all
git checkout PyGEM_RT
```
##### PyGEM-script
```
git clone git@github.com:Ruitangtang/PyGEM-scripts.git PyGEM_All
git fetch -all
git checkout FA_AdaPBS_RT
```

### Model input
- The model inputs follow the configuration used in [PyGEM](https://pygem.readthedocs.io/en/latest/model_inputs.html#model-input-table-target),with the addition of annual time series of terminus position change for tidewater glaciers in Svalbard during the period 2000-2020 from [Li et al.,2024](https://doi.org/10.5194/essd-16-919-2024).

### Model Test
- The test is for the single glacier test: the Sabinebreen Glacier, Svalbard (RGI60-07.00036)
- Model input dataset sample
- all the input data should be under the "Input" folder
- the output will be under "Output" folder

#### calibration test
```
 python -u run_calibration_AMIS_MB_FA_20002010_Parallel_Log_New.py -rgi_region01 07 -rgi_glac_number '7.00036' -ref_startyear 2000 -ref_endyear 2019 -frontalablation_fn "frontal_ablation_obs_20002010.csv" -hugonnet_fn "mass_balance_obs_20002010.csv" -lengthchange_annual_fn "lengthchange_annual_rgi_region01_7_20002020.csv" -store_monthly_step -Visualize_Index -v -debug

```
#### simulation test 

```
python -u run_simulation_AMIS_SERMeQ.py -option_parallels -rgi_region01 7 -rgi_glac_number '7.00036'  -gcm_startyear 2000 -gcm_endyear 2100 -gcm_name='NorESM2-MM' -scenario='ssp126' -hugonnet_fn "mass_balance_obs_20002010.csv" -debug
```



# PyGEM-scripts
Python and other scripts that are used for running calibration, simulation, post-processing, etc.  These scripts are meant to be used with the PyGEM repository (https://github.com/drounce/PyGEM) that has all the classes and functions and can be installed via PyPI.  Note that some of the scripts in this repository are hard-coded to specific output or datasets and thus may contain additional scripts that are not required for the typical user. The goal is to remove these over time such that all codes will work for any user; however, this is a work in progress. The primary scripts that we suggest using will be described in this here.

The Python Glacier Evolution Model (PyGEM) is an open-source glacier evolution model coded in Python that models the transient evolution of glaciers. Each glacier is modeled independently using a monthly timestep. PyGEM has a modular framework that allows different schemes to be used for model calibration or model physics (e.g., climatic mass balance, glacier dynamics).  In the newest version under development, PyGEM is working to become compatible with the Open Global Glacier Model (OGGM; https://oggm.org/).

Manual: Details concerning the model physics, installation, and running the model may be found here: https://github.com/drounce/PyGEM/wiki; however, given the rapid pace of development at present, please contact the lead developer (David Rounce) for additional documents as we will be updating the wiki soon.

Usage: PyGEM is meant for large-scale glacier evolution modeling.  PyGEM is still under active development.  Therefore, if you would like to run the model independently, it is suggested to install Release PyGEMv0.2.0.  However, given the major changes to the code, this release of the code is not being actively supported anymore.  We therefore highly encourage you to contact the lead developer (David Rounce) if you're interested in using the version that is actively being developed.

Contributing: We welcome contributions from any interested parties and are in the process of outlining how to best incorporate outside contributions. For the time being, if you would like to contribute to the development of the model, please contact David Rounce (drounce@cmu.edu).

Credits: If using PyGEM for scientific applications, please cite the following:
Rounce, D.R., Hock, R., Maussion, F., Hugonnet, R., Kochtitzky, W., Huss, M., Berthier, E., Brinkerhoff, D., Compagno, L., Copland, L., Farinotti, D., Menounos, B., and McNabb, R.W. “Global glacier change in the 21st century: Every increase in temperature matters”, Science, 379(6627), pp. 78-83, (2023), doi:10.1126/science.abo1324.
