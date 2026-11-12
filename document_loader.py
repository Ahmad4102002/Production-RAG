import time 
import os
import tempfile
from pathlib import Path 
from langchain_community.document_loaders import (PyPDFLoader)



loader = PyPDFLoader("testdata/acme_support_guide.pdf")
docs = loader.load()











