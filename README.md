# Sensitivity Analysis Toolkit
This repo aims at implementing useful tools to assess the correlation between physical model (ODE systems, PDEs) parameters. This is achieved through a self-hosted streamlit application that allows the user to visualise the local correlation matrix for a specific model (up to now, the Sestak-Berggren model) and varies the parameters at which the matrix is computed.

## Execution

Create a virtual environment
```{python}
python3 -m venv .venv
source ${PWD}/.venv/bin/activate
```

Install the necessary dependencies within the environment
```{python}
pip install -r requirements.txt
```

Finally, run the application
```{bash}
python -m streamlit run sb_correlation.py
```

## Future developments
- make the model definition customizable, using third-party mathematical engine to input the ODE model of interest
- add correlation computation for PDEs
  
