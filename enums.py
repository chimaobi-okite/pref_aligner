from enum import Enum
import os

class PrefType(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"

# Example input (replace with your actual source)
pref_type_input = "relevant"  # or "irrelevant"

#
