"""## 1. Install and mount

Colab ships torch; we only need `transformers`. Drive is mounted to reach the
fixed splits already uploaded to `My Drive/opspilot/`."""

get_ipython().system('pip install -q transformers')

import os, io, json, time, zipfile, threading, random, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
WORK = Path('/content/jev_benchmark')
WORK.mkdir(exist_ok=True)
print('mounted')
