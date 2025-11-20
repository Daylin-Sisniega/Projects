# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 23:10:57 2025

@author: dayli
"""

# config.py

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    MISTRAL_MODEL_NAME = os.getenv("MISTRAL_MODEL_NAME")
    
    DATA_FOLDER = "data"
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "rag_301445010"