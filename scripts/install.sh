#!/bin/bash


# install the rest of the dependencies
pip install -r requirements.txt

# install vllm without dependencies
pip install vllm --no-deps
