# config.py
import asyncio
import os
import random
import time
import sys
import requests
import json
import warnings
import heapq
import numpy as np
import ast
import inspect
import copy
import hashlib
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple, Set, Callable, Union
from abc import ABC, abstractmethod
import traceback

# SYSTEM CONFIGURATION
warnings.filterwarnings("ignore")

SAVE_FILE = "prometheus_genome.json"
MEMORY_FILE = "prometheus_memory.npz"
ARCHITECTURE_FILE = "prometheus_architecture.json"
SKILLS_FILE = "prometheus_skills.json"
HEALTH_FILE = "prometheus_health.json"
COMMS_FILE = "communication.txt"
TRACE_FILE = "engine_trace.log"

MEMORY_MAX_SIZE = 10000
MAX_GOAL_QUEUE_SIZE = 50

# GPU Support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    GPU_AVAILABLE = False
    xp = np

