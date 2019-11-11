#!/usr/bin/env bash

# Install TensorFlow using Anaconda

# Create a conda environment in which to install TensorFlow- strongly recommended
# TensorFlow is being updated constantly and is not yet stable, so new releases break
# old code
# Using environments keeps new installations from messing with old ones
conda create -n tensorflow python=3.6 ipython=6

# Activate the environment
source activate tensorflow

# Install tensorflow and pandas
pip install --upgrade tensorflow pandas

# You should now be able to open an ipython shell and type "import tensorflow as tf"

