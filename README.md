# AI Assistant for Data Science
This repository contains the code for an AI Assistant integrated in a Jupyter Notebook

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Features
* You can make a generic question to the LLM
* You can ask to analyze, for example, a dataset/dataframe loaded
* You can ask for the code to make some processing, for example to train a ML model

## Usage
It can be plugged-in by simply loading an IPython extension

```
%load_ext oci_genai_magics
```

then:
* %ask: make a question to the LLM
* %ask_data: ask to analyze a dataset loaded in the NB 
* %ask_code: ask to generate python code to analyze or process data

## Setup and Configuration
Configure security: for example, create a key-pair and put in $HOME/.oci

Setup OCI **policies** to enable access to OCI Generative AI

If you want to run locally, create a conda environment
```
conda create -n ai-assistant-env python==3.11

pip install -r requirements.txt
```

Put in the **config.py** file
* MODEL_ID
* Type of OCI auth to use
* Service endpoint
* OCID of your OCI compartment

## Setup in OCI Data Science
Create a NB session, 2 ECPU is OK.

Install a conda environment with Python 3.11 and activate it

```
odsc conda install -s generalml_p311_cpu_x86_64_v1
conda activate /home/datascience/conda/generalml_p311_cpu_x86_64_v1
```

Install additional Libraries
```
pip install oci -U
pip install langchain -U
pip install langchain-community -U
pip install tiktoken
```

clone the repository
```
git clone https://github.com/luigisaetta/ai-assistant-4-datascience.git
```

## Generative AI customizations
If you want to change the prompts used, they're in 
[prompts.py](https://github.com/luigisaetta/ai-assistant-4-datascience/blob/main/prompts.py)

# Libraries used:
* OCI Python SDK
* Langchain
* Langchain-community

## Suggestions for Data Analysis
The AI Assistant can give suggestions to improve Data Analysis, but you shouldn't rely only on these suggestions.
**Always** check the results from the LLM.

There are several reasons why analysis from the LLM cannot be completely reliable:
1. Lack of Specialized Analytical Capabilities: LLMs are designed to process and generate human-like text but are not inherently equipped with the statistical and analytical tools necessary for complex data analysis. Traditional data analysis tools and statistical models are often more effective for tasks requiring specific analytical computations.
2. Potential for Hallucinations and Inaccuracies: LLMs can produce outputs that appear plausible but are factually incorrect or nonsensical, a phenomenon known as "hallucination." This unreliability poses significant challenges when precise data analysis and accurate predictions are required. 
3. Lack of Interpretability: LLMs operate as black boxes, providing little insight into how they arrive at specific outputs. In data analysis, understanding the rationale behind predictions is crucial for validation and decision-making, a transparency that LLMs do not inherently offer.

In addition, for big datasets only a sample is passed in the context of the request to the LLM. See:
* MAX_ROWS_IN_SAMPLE in config

The AI assistant can be a good **assistant** for example to suggest you **Python code**. Try it!

## Next steps
There is an official integration for Generative AI models in Jupyter: [jupyter-ai](https://github.com/jupyterlab/jupyter-ai)

I' working on developing an integration with jupyter-ai, stay tuned and... let me know if you're interested.

