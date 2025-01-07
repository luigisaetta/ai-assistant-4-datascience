"""
OCI models configuration
"""

# MODEL_ID = "cohere.command-r-plus-08-2024"
MODEL_ID = "meta.llama-3.1-70b-instruct"

AUTH = "API_KEY"
SERVICE_ENDPOINT = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"

TEMPERATURE = 0.1
MAX_TOKENS = 1024

# OCI general
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaushuwb2evpuf7rcpl4r7ugmqoe7ekmaiik3ra3m7gec3d234eknq"

# history management
MAX_MSGS_IN_HISTORY = 10
# context.py
# Maximum number of rows to display in a sample
MAX_ROWS_IN_SAMPLE = 500
