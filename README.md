# NESCQR

This repository is the implementation of the paper: A novel wind power interval prediction method based on neural ensemble search and dynamic conformalized quantile regression (<b>NESCQR</b>). 

## Quick start
### 1. Install dependencies 
Install the environment dependencies using the following command:

```bash
conda create -n nescqr python=3.9
conda activate nescqr
pip install -r requirements.txt
```

If the `torch` version in  `requirements.txt`  is incompatible with your device, you can install the dependencies using `pip install -r requirements_no_torch.txt`, and then visit the [PyTorch official website](https://pytorch.org/get-started/locally/) to choose the appropriate version of torch for installation.

### 2. Install latex and add it to the system's PATH variable
To support `scienceplots` library, you need to install LaTeX software on your device and add the installation path to the system's PATH variable. 

Here's an example for Windows. It's recommended to install [MikTeX](https://miktex.org/download), and then add the MikTeX installation path to the system's environment variables. An example of system environment variables is shown in the following figure:

![windows_miketx_path_variable](./images/windows_miketx_path_variable.png)


### 3. Download and extract data
Download the datasets into the `./NESCQR/data/` folder. Below are the two public datasets used in this study:

[1] [Kaggle: Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting/) <br>
[2] [Kaggle: Wind Turbine Power (kW) Generation Data](https://www.kaggle.com/datasets/psycon/wind-turbine-energy-kw-generation-data/data) <br>

After downloading, unzip these two datasets into the `'./NESCQR/data/'` directory.

In addition, `International Spanish Data.csv` and `yumen.csv` are two private datasets used in this paper.

### 4. Run
It is recommended to use a GPU for running the experiments. You can modify the parameters in the `config.json` file.

Run the experiments on the wind power dataset with the command:
`python main_wind_power.py`

The same as the other datasets:
`python main_wind_turbine.py` <br>
`python main_Spanish.py` <br>
`python main_Yumen.py` <br>

Once the execution is complete, you can find the results for all methods in the `./NESCQR/result/` folder.
