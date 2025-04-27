from enum import Enum
import os

class PrefType(str, Enum):
    RELEVANT = "relevant"
    IRRELEVANT_SET = "irrelevant_set"
    IRRELEVANT = "irrelevant"
    
class PromptMethod(str, Enum):
    ICL = "icl"
    DIRECT = "direct"
    COT = "cot"

# Example input (replace with your actual source)
pref_type_input = "relevant"  # or "irrelevant"

#
