#CREATE VIRTUAL ENVIRONMENT IF NOT ALREADY DONE
#CD TO DIRECTORY FOR PROJECT
python3 -m venv env

#START VIRTUAL ENVIRONMENT
source env/bin/activate
which python

#PIP INSTALL IPYKERNEL
pip install ipykernel

#PYTHON INSTALL IPYKERNEL TO VIRTUAL ENVIRONMENT
python -m ipykernel install --user --name=name of virtual environment

#START JUPYTER NOTEBOOK
jupyter notebook
