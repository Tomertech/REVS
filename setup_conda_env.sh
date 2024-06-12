#!/bin/bash

# Define the YAML file name
YML_FILE="revsenv.yml"

# Check if the YAML file exists
if [ ! -f "$YML_FILE" ]; then
    echo "Error: File '$YML_FILE' not found."
    exit 1
fi

# Extract environment name from the YAML file
ENV_NAME=$(grep "name: " $YML_FILE | cut -d " " -f 2)

# Check if environment name was extracted
if [ -z "$ENV_NAME" ]; then
    echo "Error: Could not extract environment name from $YML_FILE."
    exit 1
fi

# Check if the conda environment already exists
if conda info --envs | grep "^$ENV_NAME\s" > /dev/null; then
    echo "Conda environment '$ENV_NAME' already exists. Do you want to recreate it? (y/n)"
    read REPLY
    if [ "$REPLY" != "y" ]; then
        echo "Exiting without creating environment."
        exit 0
    fi
    # Remove the existing environment
    echo "Removing existing environment '$ENV_NAME'."
    conda env remove -n $ENV_NAME
fi

# Create the new environment from the YAML file
echo "Creating new conda environment '$ENV_NAME' from $YML_FILE."
conda env create -f $YML_FILE

echo "Environment '$ENV_NAME' created successfully."
