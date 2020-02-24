# PAWS
Protection Assistant for Wildlife Security (PAWS). Prediction and prescription to combat illegal wildlife poaching. Teamcore Research Group, Harvard University.

The following students have contributed to this codebase and methodology: Fei Fang, Benjamin Ford, Shahrzad Gholami, Debarun Kar, Laksh Matai, Sara Mc Carthy, Thanh H. Nguyen, Lily Xu

## Directories
- `./preprocess/` - R code for processing raw data
- `./preprocess_consolidate/` - Python code for consolidating output from preprocessing
- `./iware/` - Python code for ML risk prediction
- `./planning/` - Python code for patrol planning. (Note that this also includes older versions of prediction code.)
- `./field_tests/` - Python code for selecting field test areas and visualizing park geography and risk maps

These directories must be made:
- `./inputs/` - raw data: CSV file of patrol observations and shapefiles of geographic features
- `./outputs/` - output of preprocessing step

## Processing order
In `./preprocess/`, execute the `pipeline` script to run all required preprocessing steps.

In `./preprocess_consolidate/`, execute the driver script, which will call all necessary functions.

In `./iware/`, execute the driver script, and choose whether to test (to run train/test code and evaluate performance) or make predictions. Run `visualize_maps.py` to generate images of the riskmaps.

In `./planning/`, follow the README there.

In `./field_tests/`, execute `select_field_test.py` to run relevant tests

## Relevant papers

**Stay Ahead of Poachers: Illegal Wildlife Poaching Prediction and Patrol Planning Under Uncertainty with Field Test Evaluations**

Lily Xu, Shahrzad Gholami, Sara Mc Carthy, Bistra Dilkina, Andrew Plumptre, Milind Tambe, Rohit Singh, Mustapha Nsubuga, Joshua Mabonga, Margaret Driciru, Fred Wanyama, Aggrey Rwetsiba, Tom Okello, Eric Enyel

Proceedings of the 36th IEEE International Conference on Data Engineering (ICDE) 2020

https://arxiv.org/abs/1903.06669

    @inproceedings{xu2020stay,
     author = {Xu, Lily and Gholami, Shahrzad and Mc Carthy, Sara and Dilkina, Bistra and  Plumptre, Andrew and Tambe, Milind and Singh, Rohit and Nsubuga, Mustapha and Mabonga, Joshua and Driciru, Margaret and Wanyama, Fred and Rwetsiba, Aggrey and Okello, Tom and Enyel, Eric},
     title = {Stay Ahead of Poachers: Illegal Wildlife Poaching Prediction and Patrol Planning Under Uncertainty with Field Test Evaluations},
     booktitle = {Proceedings of the 36th IEEE International Conference on Data Engineering (ICDE)},
     year = {2020}
    }


**Adversary models account for imperfect crime data: Forecasting and planning against real-world poachers**

Shahrzad Gholami, Sara Mc Carthy, Bistra Dilkina, Andrew Plumptre, Milind Tambe, Margaret Driciru, Fred Wanyama, Aggrey Rwetsiba, Mustapha Nsubaga, Joshua Mabonga, Tom Okello, Eric Enyel

Proceedings of the 17th International Conference on Autonomous Agents and Multi-agent Systems (AAMAS) 2018

https://projects.iq.harvard.edu/files/teamcore/files/2018_28_teamcore_sgholami_aamas18.pdf

    @inproceedings{gholami2018adversary,
      title={Adversary models account for imperfect crime data: Forecasting and planning against real-world poachers},
      author={Gholami, Shahrzad and Mc Carthy, Sara and Dilkina, Bistra and Plumptre, Andrew and Tambe, Milind and Driciru, Margaret and Wanyama, Fred and Rwetsiba, Aggrey and Nsubaga, Mustapha and Mabonga, Joshua and others},
      booktitle={Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems},
      pages={823--831},
      year={2018},
      organization={International Foundation for Autonomous Agents and Multiagent Systems}
    }

**Cloudy with a chance of poaching: Adversary behavior modeling and forecasting with real-world poaching data**

Debarun Kar, Benjamin Ford, Shahrzad Gholami, Fei Fang, Andrew Plumptre, Milind Tambe, Margaret Driciru, Fred Wanyama, Aggrey Rwetsiba, Mustapha Nsubaga, Joshua Mabonga

Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (AAMAS) 2017

    @inproceedings{kar2017cloudy,
      title={Cloudy with a chance of poaching: Adversary behavior modeling and forecasting with real-world poaching data},
      author={Kar, Debarun and Ford, Benjamin and Gholami, Shahrzad and Fang, Fei and Plumptre, Andrew and Tambe, Milind and Driciru, Margaret and Wanyama, Fred and Rwetsiba, Aggrey and Nsubaga, Mustapha and others},
      booktitle={Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems},
      pages={159--167},
      year={2017},
      organization={International Foundation for Autonomous Agents and Multiagent Systems}
    }

https://projects.iq.harvard.edu/files/teamcore/files/2017_5_teamcore_aamas2017_intercept.pdf


**Deploying PAWS: Field optimization of the protection assistant for wildlife security**

Fei Fang, Thanh H. Nguyen, Rob Pickles, Wai Y. Lam, Gopalasamy R. Clements, Bo An, Amandeep Singh, Milind Tambe, Andrew Lemieux

IAAI 2016

https://projects.iq.harvard.edu/files/teamcore/files/2016_4_teamcore_iaai16_paws.pdf

    @inproceedings{fang2016deploying,
      title={Deploying PAWS: Field optimization of the protection assistant for wildlife security},
      author={Fang, Fei and Nguyen, Thanh H and Pickles, Rob and Lam, Wai Y and Clements, Gopalasamy R and An, Bo and Singh, Amandeep and Tambe, Milind and Lemieux, Andrew},
      booktitle={Twenty-Eighth IAAI Conference},
      year={2016}
    }
