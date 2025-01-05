# AI Assistant for Data Science
This repository contains the code to develop a prototype of an AI Assistant integrated in a Jupyter Notebook

## Features
* You can make a generic question to the LLM
* You can ask to analyze, for example, a dataframe loaded
* You can ask for the code to make some processing, for example to train a ML model

## Usage
It can be plugged by simply loading an IPython extension

```
%load_ext oci_genai_magics
```

then:
* %ask: make a question to the LLM
* %ask_data: ask to analyze a dataset loaded in the NB 
* %ask_code: ask to generate python code to analyze or process data

## Setup and Configuration
Configure security: for example, create a key-pair and put in $HOME/.oci

Setup in OCI **policies** to enable access to OCI Generative AI

If you want to run locally, create a conda environment
```
conda create -n ai-assistant-env python==3.11

pip install -r requirements.txt
```

Put in the **config.py** file
* MODEL_ID
* type of OCI auth to use
* service endpoint
* OCID of your OCI compartments

## Generative AI customizations
If you want to change the prompts used, they're in **prompts.py**

# Libraries used:
* OCI Python SDK
* Langchain
* Langchain-community
