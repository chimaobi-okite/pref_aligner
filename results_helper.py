from calendar import c
from functools import partial
import os
import glob
import itertools
from re import A
from tkinter import font

from matplotlib.font_manager import font_scalings
from matplotlib.pylab import f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, patches
import matplotlib.colors as mcolors
import seaborn as sns

FONT_SIZE = 15
TICK_SIZE = 15
# LEGEND_FONT_SIZE = 15
LEGEND_FONT_SIZE = 20

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=TICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=42)  # fontsize of the figure title
plt.rcParams["font.family"] = "Times New Roman"

# -----------------------------
# Global variables
# -----------------------------

RELEVANCE = [
    "relevant",
    "irrelevant_set",
    "irrelevant",
]
RELEVANCE_DICT = {
    "relevant": "Relevant",
    "irrelevant_set": "Mixed",
    "irrelevant": "Irrelevant",
}
# Prompt methods to process
PROMPT_METHODS = [
    "direct",
    "cot",
    "icl",
    "self_critic",
    "aligner"
]
PROMPT_METHODS_WITHOUT_ALIGNER = [
    "direct",
    "cot",
    "icl",
    "self_critic"
]
PROMPT_METHODS_DICT = {
    "direct": "Zero-Shot",
    "cot": "CoT",
    "icl": "ICL",
    "self_critic": "Self-Critic",
    "aligner": "Pref-Aligner"
}

# List of models (as provided)
MODELS_FULL = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "kaist-ai/janus-7b",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-4o-mini-2024-07-18",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "qwen/Qwen3-8B",
    "qwen/Qwen3-8B_thinking",
    "qwen/Qwen3-32B",
    "qwen/Qwen3-32B_thinking",
]
# Process model names
MODELS = [model.split("/")[-1] for model in MODELS_FULL]
MODELS_SHORT = [
    "Llama3-8B",
    "Llama3-70B",
    "Mistral-7B",
    "Janus-7B",
    "Mixtral-8x7B",
    "GPT-4o-mini",
    "DeepSeek-R1-70B",
    "Gemma-2-9B",
    "Gemma-2-27B",
    "Qwen3-8B",
    "Qwen3-8B-Thinking",
    "Qwen3-32B",
    "Qwen3-32B-Thinking",
]
MODELS_SHORT_DICT = {MODELS[i]: MODELS_SHORT[i] for i in range(len(MODELS))}

# The input dataset directories (each should contain a CSV per model)
DATASETS_FULL = [
    "tau/commonsense_qa",
    "cais/mmlu",
    "truthfulqa/truthful_qa",
]
# Process dataset names
DATASETS = [dataset.split("/")[-1] for dataset in DATASETS_FULL]

DATASET_SHORT_DICT = {
    "commonsense_qa": "CQA",
    "mmlu": "MMLU",
    "truthful_qa": "TQA",
}

DATASET_LENS = {
    "commonsense_qa": 1221,
    "mmlu": 5170,
    "truthful_qa": 817,
    "tau/commonsense_qa": 1221,
    "cais/mmlu": 5170,
    "truthfulqa/truthful_qa": 817,
}

# The target output subdirectory (under each prompt method) where aggregated results are saved.
TARGET_DATASET = "full"
# The robust dataset to use for aggregation
ROBUST_PATH = os.path.join("data", "robuset.csv")

from preferences.prefs import MMLU_PREFS, TQA_PREFS, CQA_PREFS

DATASET_PREFERENCES = {
  "tau/commonsense_qa": CQA_PREFS,
  "cais/mmlu": MMLU_PREFS,
  "truthfulqa/truthful_qa": TQA_PREFS,
  "commonsense_qa": CQA_PREFS,
  "mmlu": MMLU_PREFS,
  "truthful_qa": TQA_PREFS,
}

CQA_PREF_DICT = {s[:30]:i for i, s in enumerate(CQA_PREFS.values())}
MMLU_PREF_DICT = {s[:30]:i for i, s in enumerate(MMLU_PREFS.values())}
TQA_PREF_DICT = {s[:30]:i for i, s in enumerate(TQA_PREFS.values())}
PREF_DICTS = {
    "tau/commonsense_qa": CQA_PREF_DICT,
    "cais/mmlu": MMLU_PREF_DICT,
    "truthfulqa/truthful_qa": TQA_PREF_DICT,
    "commonsense_qa": CQA_PREF_DICT,
    "mmlu": MMLU_PREF_DICT,
    "truthful_qa": TQA_PREF_DICT,
}

METRICS = ["BR", "RER", "AFR", "PVR"]
METRIC_LABELS = {
    "BR": "Breakage Rate",
    "RER": "Robustness Error",
    "AFR": "Alignment Failure",
    "PVR": "Performance Variation",
}

METRIC_LABELS_ABBR = {
    "BR": "BR",
    "RER": "RER",
    "AFR": "AFR",
    "PVR": "PVR",
}

class MultiKeyDict:
    def __init__(self, keys, cat_keys={}):
        # Defaults to some custom keys and a field entry
        if "field" in keys:
            raise ValueError("Key 'field' is reserved and cannot be used.")
        self.custom_keys = keys + ["field"]
        self.fields = set()
        self.data = {}
        self.cat_keys = cat_keys
        for varkey in cat_keys.keys():
            if varkey not in self.custom_keys:
                raise ValueError(f"Key '{varkey}' is not in the initialized keys.")
            try:
                klen = len(cat_keys[varkey])
                assert klen > 0
            except TypeError:
                raise ValueError(f"Key '{varkey}' must have positive finite length")

    def _get_key(self, **kwargs):
        # Ensure the key order follows the initialized keys
        return tuple(kwargs.get(k, None) for k in self.custom_keys)

    def add(self, data, **kwargs) -> None:
        # Ensure all keys are specified and no extras
        provided = set(kwargs.keys())
        required = set(self.custom_keys)
        missing = required - provided
        extra = provided - required
        if missing:
            raise ValueError(f"Missing keys for add: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected keys for add: {sorted(extra)}")
        # Add field to self.fields if not already present
        field = kwargs["field"]
        if field not in self.fields:
            self.fields.add(field)

        key = self._get_key(**kwargs)
        self.data[key] = data

    def get(self, **kwargs):
        # Allow querying multiple values (list/tuple/set) per key
        # Ensure 'field' is specified
        if 'field' not in kwargs:
            raise ValueError("Field must be specified in the query")

        results = {}
        for key, value in self.data.items():
            match = True
            for idx, dim in enumerate(self.custom_keys):
                q = kwargs[dim] if dim in kwargs else None
                if q is None:
                    continue
                # If list/tuple/set (but not str), match any
                if isinstance(q, (list, tuple, set)) and not isinstance(q, str):
                    if key[idx] not in q:
                        match = False
                        break
                else:
                    if key[idx] != q:
                        match = False
                        break
            if match:
                # print("Adding", key)
                results[key] = value
        return results

    def keys(self, **kwargs):
        return self.get(**kwargs).keys()

    def values(self, **kwargs):
        return self.get(**kwargs).values()

    def items(self, **kwargs):
        return self.get(**kwargs).items()
    
    def get_field_matrix(
        self,
        field,
        fixed_keys,
        axis_keys=None,
        fill_missing=None
    ):
        """
        Construct N-dimensional matrices for one or multiple fields,
        fixing some keys and varying others as axes. The first dimension
        will be the field(s) specified.

        Args:
            field: A field name or list/tuple/set of field names.
            fixed_keys: Mapping of dimension keys to fixed values.
            axis_keys: Optional list of dimension keys to use as matrix axes; if None,
                       deduced as all keys not in fixed_keys.
            fill_missing: Value to use when a combination has no entry.

        Returns:
            A tuple (matrix, axis_keys_out). matrix is a nested list where dimension 0
            indexes fields, followed by axes in axis_keys.
            axis_keys_out is ['field'] + axis_keys.
        """
        # Determine valid dimension keys (exclude 'field')
        dims = [k for k in self.custom_keys if k != "field"]

        # Validate fixed_keys
        for k in fixed_keys:
            if k not in dims:
                raise ValueError(f"Fixed key '{k}' is not a valid dimension.")

        # Deduce axis_keys if not provided
        if axis_keys is None:
            axis_keys = [k for k in dims if k not in fixed_keys]
        else:
            for k in axis_keys:
                if k not in dims:
                    raise ValueError(f"Axis key '{k}' is not a valid dimension.")
                if k in fixed_keys:
                    raise ValueError(f"Axis key '{k}' cannot also be in fixed_keys.")

        # Ensure coverage of all dimensions
        if set(fixed_keys) | set(axis_keys) != set(dims):
            missing = set(dims) - (set(fixed_keys) | set(axis_keys))
            raise ValueError(f"Keys must cover all dimensions; missing: {missing}")

        # Prepare list of fields
        if isinstance(field, (list, tuple, set)) and not isinstance(field, str):
            fields_list = list(field)
        else:
            fields_list = [field]

        # Validate fields
        for f in fields_list:
            if f not in self.fields:
                raise ValueError(f"Field '{f}' not present in any entries.")

        # Recursive builder for a single field
        def build_for(f):
            def build_matrix(dim, current_vals):
                if dim == len(axis_keys):
                    query = {**fixed_keys}
                    for i, ak in enumerate(axis_keys):
                        query[ak] = current_vals[i]
                    query['field'] = f
                    result = self.get(**query)
                    return next(iter(result.values()), fill_missing)
                else:
                    key = axis_keys[dim]
                    vals = self.cat_keys.get(key)
                    if vals is None:
                        raise ValueError(f"No categories for axis key '{key}'")
                    return [build_matrix(dim + 1, current_vals + [val])
                            for val in vals]
            return build_matrix(0, [])

        # Build full matrix with field as outermost dimension
        full_matrix = [build_for(f) for f in fields_list]
        # Prepend 'field' to axis keys for output
        axis_keys_out = ['field'] + axis_keys
        return full_matrix, axis_keys_out

    def __len__(self):
        return len(self.data)

    def _data_repr(self, key):
        val = self.data[key]
        if hasattr(val, "shape"):
            return f"{type(val).__name__} of shape {val.shape}"
        elif hasattr(val, "__len__") and not isinstance(val, str):
            return f"{type(val).__name__} of len {len(val)}"
        else:
            return str(val)

    def __str__(self):
        repr = f"MultiKeyDict with {len(self.data)} entries:\n"
        repr += f"Custom Keys: {self.custom_keys}\n"
        repr += "Categorical Fields:\n"
        for k, v in self.cat_keys.items():
            repr += f"  - {k}({len(v)}): {v}\n"
        repr += f"  - Fields(len {len(self.fields)}): {list(self.fields)}\n"
        return repr

    def print_data(self):
        print("Data:")
        for key in self.data.keys():
            print(f"  - {key}: {self._data_repr(key)}")

#----------------------------------
# 
# Aggregate dataset results
# 
#----------------------------------
# First, verify the robust dataset exists.
if not os.path.exists(ROBUST_PATH):
    print(f"Robust dataset not found at {ROBUST_PATH}. Exiting.")
    exit(1)
robust_df = pd.read_csv(ROBUST_PATH)
robust_rows = len(robust_df)
# Create stats folder and compute stats
stats_folder = os.path.join("results", "mcq_results", "relevant","stats")
os.makedirs(stats_folder, exist_ok=True)

def make_file_path(relevance, method, dataset, model, use_eval=True, long=False):
    file_name = f"{model}{f'-{method}-{dataset}' if long else ''}{('_eval' if use_eval else '')}.csv"
    return os.path.join("results", "mcq_results", relevance, method, dataset, file_name)

def make_file_path_tmpl(relevance, method, dataset, model, use_eval=True):
    file_name = f"{model}*{('_eval' if use_eval else '')}.csv"
    return os.path.join("results", "mcq_results", relevance, method, dataset, file_name)

def get_file(relevance, method, dataset, model, use_eval=True, verbose=False):
    tmpl = make_file_path_tmpl(relevance, method, dataset, model, use_eval=use_eval)
    files = list(glob.glob(tmpl))
    if len(files) == 0:
        if verbose:
            print(f"Warning: No files found for [{relevance}, {method}, {dataset}, {model}]. Check combination")
        return None
    if not use_eval:
        files = [f for f in files if not f.endswith("_eval.csv")]
    if len(files) > 1 and verbose:
        print(f"Warning: Multiple files found for [{relevance}, {method}, {dataset}, {model}]. Check combination")
    return files[0]

def validate_full_dataset(relevance, method, model, use_eval, verbose=False):
    """
    Validate that full dataset
    """
    # Get file nmae
    goal_dataset_path = get_file(relevance, method, TARGET_DATASET, model, use_eval=use_eval)
    if goal_dataset_path is None:
        if verbose:
            print(f"Warning: No files found for [{relevance}, {method}, {TARGET_DATASET}, {model}]. Check combination")
        return False
    exist_df = pd.read_csv(goal_dataset_path)
    exist_rows = len(exist_df)
    # True if have the same column names and same number of rows
    return (exist_rows == robust_rows)

def extend_irrelevant(filename):
    df = pd.read_csv(filename)
    # Compute column "followed_pref" as all 0.0
    # Compute column "is_robust" as nopref_correct
    df["followed_pref"] = 0.0
    df["is_robust"] = (df["pref_answer"] == df["gold_option"])
    df["pref_ans"] = df["pref_answer"]
    df["no_pref_ans"] = df["nopref_answer"]
    # Save file from {filename}.csv to {filename}-aligner-full.csv
    new_filename = filename.replace(".csv", "-aligner-full_eval.csv")
    df.to_csv(new_filename, index=False)

def compute_non_eval(dataset, data_df):
    """
    Deprecated: compute non-eval dataset extra columns for earlier in experiments
    """
    # Get chunk from robust_df
    robust_chunk = robust_df[robust_df["source"] == dataset].copy()
    assert len(data_df)==len(robust_chunk), "Sizes don't match!"

    data_df = data_df.sort_values(by=["question", "options"]).reset_index(drop=True)
    robust_chunk = robust_chunk.sort_values(by=["question", "options"]).reset_index(drop=True)
    # Check size
    assert len(data_df)==len(robust_chunk), "Sizes don't match!"

    # Assert correct answers
    data_correct_ans = data_df["gold_option"].reset_index(drop=True)
    robust_correct_ans = robust_chunk["gold_option"].reset_index(drop=True)
    assert len(data_correct_ans)==len(robust_correct_ans), "Option Sizes don't match!"

    pref_library = DATASET_PREFERENCES.get(dataset, {})
    # Reverse the TQA_PREFS dictionary: keys become values and vice versa.
    rev_prefs = {v: k for k, v in pref_library.items()}

    # create a new column "pref_index" that maps the text to its corresponding index.
    robust_chunk["pref_index"] = robust_chunk.loc[:,"preference"].map(rev_prefs)
    pref_text = robust_chunk.apply(
        lambda row: pref_library[int(row['pref_index'])],
        axis=1
    )
    pref_res = robust_chunk.apply(
        lambda row: data_df.loc[row.name, f"profile_{int(row['pref_index'])}_res"],
        axis=1
    )
    pref_answer = robust_chunk.apply(
        lambda row: data_df.loc[row.name, f"profile_{int(row['pref_index'])}_answer"],
        axis=1
    )
    pref_correct = pref_answer.eq(data_df["gold_option"])
    nopref_res = robust_chunk.apply(
        lambda row: data_df.loc[row.name, f"profile_0_res"],
        axis=1
    )
    nopref_answer = robust_chunk.apply(
        lambda row: data_df.loc[row.name, f"profile_0_answer"],
        axis=1
    )
    nopref_correct = nopref_answer.eq(data_df["gold_option"])
    # Start building a new dataframe
    new_df_dict = {
        "question": data_df["question"],
        "options": data_df["options"],
        "gold_option": data_df["gold_option"],
        "source": robust_chunk["source"],
        "profile_idx": robust_chunk["pref_index"],
        "preference": pref_text,
        "pref_res": pref_res,
        "pref_answer": pref_answer,
        "pref_correct": pref_correct,
        "nopref_res": nopref_res,
        "nopref_answer": nopref_answer,
        "nopref_correct": nopref_correct,
    }
    return pd.DataFrame(new_df_dict)

def compute_full(full_df):
    """
    Deprecated: compute full dataset extra columns for earlier in experiments
    """
    if 'nopref_correct' not in full_df.columns:
        full_df['nopref_correct'] = full_df["gold_option"] == full_df["no_pref_ans"]
    if 'pref_correct' not in full_df.columns:
        full_df['pref_correct'] = full_df["gold_option"] == full_df["pref_ans"]
    return full_df

def aggregate_instance(file_names,
        relevance,
        method,
        model,
        use_eval=False,
        force_reaggregate=False,
        verbose=False
    ):
    """
    Aggreggate multiple dataset-specific results to full result
    """
    goal_path = make_file_path(
        relevance, method, TARGET_DATASET, model, use_eval=use_eval, long=True
    )
    # If not force aggregate and exists, skip
    # Aggregate files to full dataset
    processed_dfs = []
    for file_name, dataset, full_dataset in zip(file_names, DATASETS, DATASETS_FULL):
        if verbose:
            print(f"Processing {method}, {model}, {dataset}")
        data_df = pd.read_csv(file_name)
        if not use_eval:
            data_df = compute_non_eval(full_dataset, data_df)
        processed_dfs.append(data_df)
        
    # Concatenate and save df
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    assert len(combined_df)==robust_rows, "Sizes don't match!"
    combined_df.to_csv(goal_path, index=False)
    if verbose:
        print(f"Saved {method}, {model} to full")
    return goal_path
        
def check_chunk_completeness(file_names, relevance, method, model, verbose=False):
    for dataset, dataset_chunk in zip(DATASETS, file_names):
        if dataset_chunk is None:
            if verbose:
                print(f"Warning: No files found for [{relevance}, {method}, {model}, {dataset}]")
            return False
    return True

def aggregate(relevance=RELEVANCE, use_eval=False, force_reaggregate=False, verbose=False):
    # Check all files exist
    done_counter = 0
    full_counter = 0

    # Build a results dict
    keys = ["relevance", "method", "model", "dataset"]
    cat_keys = {
        "relevance": RELEVANCE,
        "method": PROMPT_METHODS,
        "model": MODELS,
        "dataset": DATASETS + ["full"]
    }
    results_dict = MultiKeyDict(
        keys, cat_keys=cat_keys
    )

    # Iterate over all relevance, methods, models, evals
    for relevance, method, model in itertools.product(relevance, PROMPT_METHODS, MODELS):
        dataset_chunks = [get_file(relevance, method, dataset, model, use_eval=use_eval) for dataset in DATASETS]
        combined_path = None
        if not force_reaggregate and \
            validate_full_dataset(relevance, method, model, use_eval, verbose=verbose):
            if verbose:
                print(f"Skipping {relevance} {method} {model} as it already exists.")
            combined_path = make_file_path(
                relevance, method, TARGET_DATASET, model, use_eval=use_eval, long=True
            )
            done_counter += 1
        # Otherwise, if all chunks exist
        elif check_chunk_completeness(dataset_chunks, relevance, method, model, verbose=verbose):
            # Aggregate
            for fpath, dataset in zip(dataset_chunks, DATASETS):
                results_dict.add(
                    fpath,
                    relevance=relevance,
                    method=method,
                    model=model,
                    dataset=dataset,
                    field="dataset_path"
                )
            combined_path = aggregate_instance(
                dataset_chunks,
                relevance,
                method,
                model,
                use_eval=use_eval,
                force_reaggregate=force_reaggregate
            )
            if verbose:
                print(f"Completed files for {relevance} {method}, {model}")
            done_counter += 1
        # Add the full dataset path
        results_dict.add(
            combined_path,
            relevance=relevance,
            method=method,
            model=model,
            dataset="full",
            field="dataset_path"
        )
        full_counter += 1
    print(f"Found {done_counter}/{full_counter} full datasets")
    return results_dict

def split_to_df_by_dataset(results_dict, verbose=False):
    """
    Build a multi-key dictionary of dataframes
    """
    
    for (relevance, method, model, _, _), full_path in results_dict.items(
        dataset="full",
        field="dataset_path",
    ):
        if full_path is None or not os.path.exists(full_path):
            if verbose:
                print(f"Warning: No full dataset found for [{relevance}, {method}, {model}].")
            continue
        full_df = pd.read_csv(full_path)
        full_df = compute_full(full_df)
        results_dict.add(
            full_df,
            relevance=relevance,
            method=method,
            model=model,
            dataset="full",
            field="df",
        )
        # Split by dataset
        for dataset, d_full in zip(DATASETS, DATASETS_FULL):
            dataset_df = None
            if full_df is not None:
                dataset_df = full_df[full_df["source"] == d_full].copy()
                assert len(dataset_df) == DATASET_LENS[dataset], f"Dataset {dataset} size mismatch"
            # Add to the dictionary
            results_dict.add(
                dataset_df,
                relevance=relevance,
                method=method,
                model=model,
                dataset=dataset,
                field="df",
            )
    return results_dict

#----------------------------------
# 
# DF Helpers & Criterion
# 
#----------------------------------

def BR(chunk, is_irrelevant=False):
    correct_no_pref_chunk = chunk.loc[chunk['nopref_correct']==1, :]
    robust_col = correct_no_pref_chunk.loc[:, "pref_correct"].mean()
    br = 1 - robust_col
    return br

def RER(chunk, is_irrelevant=False):
    if is_irrelevant:
        return 1 - chunk.loc[chunk["nopref_correct"]==1, "pref_correct"].mean()
    return 1 - chunk.loc[chunk["nopref_correct"]==1, "is_robust"].mean()

def AFR(chunk, is_irrelevant=False):
    """Updated to take nopref_correct only"""
    robust_col = chunk.loc[chunk["nopref_correct"]==1, "followed_pref"].mean()
    afr = 1 - robust_col
    # corr_pref = chunk.loc[chunk["nopref_correct"]==1, "pref_correct"].mean()
    # afr = 1 - robust_col/corr_pref
    return afr

def PVR(chunk, is_irrelevant=False):
    """Keep API consistent for error"""
    robust = chunk["pref_correct"].astype(bool)
    no_pref = chunk["nopref_correct"].astype(bool)
    # Calculate IoU
    intersection = (robust & no_pref).sum()
    union = (robust | no_pref).sum()
    # Stability
    if union == 0:
        return 0.0
    pvr = 1 - intersection / union
    return pvr

def NoPrefInvalid(chunk, is_irrelevant=False):
    """
    Computes the percentage of nopref answer that is not -1
    """
    return (chunk['nopref_answer']=='-1').mean()

def PrefInvalid(chunk, is_irrelevant=False):
    """
    Computes the percentage of pref answer that is not -1
    """
    return (chunk['pref_answer']=='-1').mean()

def MCValidDrop(chunk, is_irrelevant=False):
    """
    Computes nopref_succ = 1-(chunk["nopref_answer] == -1)/len(chunk)
    and compares with pref_succ = 1-(chunk["nopref_answer"]==-1)/len(chunk)
    returns pref_succ - nopref_succ
    """
    # Compute the number of correct answers
    no_pref_chunk = 1 - (chunk['nopref_answer']=='-1').mean()
    pref_chunk = 1 - (chunk['pref_answer']=='-1').mean()
    return pref_chunk - no_pref_chunk


def compute_metrics(results_dict, verbose=False):
    for k, df in results_dict.items(
        field="df",
    ):
        (relevance, method, model, dataset, _) = k
        for name, func in zip(
            ["BR", "RER", "AFR", "PVR", "NoPrefInvalid", "PrefInvalid", "MCValidDrop"],
            [BR, RER, AFR, PVR, NoPrefInvalid, PrefInvalid, MCValidDrop],
        ):  
            is_irrelevant=(relevance=='irrelevant')
            # print(f"Is irrelevant: {is_irrelevant} for relevance {relevance}")
            percentage = func(df, is_irrelevant) * 100. if df is not None else None
            results_dict.add(
                percentage,
                relevance=relevance,
                method=method,
                model=model,
                dataset=dataset,
                field=name,
            )
            if verbose:
                print(f"Added {name}={percentage} to", relevance, method, model, dataset)
    return results_dict


def filter_axis(data, axis, filter_val=None, labels_to_filter=None):
    assert axis < 2, "Axis must be 0 or 1"
    mask = ~np.any(data == filter_val, axis=~axis)
    if axis == 0:
        data = data[mask].astype(np.float64)
    elif axis == 1:
        data = data[:, mask].astype(np.float64)
    if labels_to_filter is not None:
        labels = [f for f, keep in zip(labels_to_filter, mask) if keep]
    else:
        labels = None
    return data, mask, labels
    
#----------------------------------
# 
# Plot results helper
# 
#----------------------------------

def print_metric_table(results_dict, relevances=RELEVANCE, metrics=["BR", "RER", "AFR", "PVR"]):
    # Compute metrics
    last_method = None
    last_relevance = None

    def print_table(arr_list, ylabels):
        if len(arr_list) == 0:
            print("None")
            return
        arr = np.stack(arr_list, axis=0)
        for i, model in enumerate(ylabels):
            row = arr[i]
            print(f"{model}\t", end="\t")
            for j in range(len(row)):
                if last_relevance == "irrelevant" and j%4==2:   # Irrelevant RER = '-'
                    print("& - ", end=" ")
                elif row[j] == min(arr[:, j]):
                    print(f"& \\textbf{{{row[j]:.1f}}}", end=" ")
                else:
                    print(f"& {row[j]:.1f}", end=" ")
            print("\\\\")
        # print("\\rowcolor{gray!15}\t")
        # print("\\textbf{Average}", end=" ")
        # mean = arr.mean(axis=0)
        # for i in range(len(mean)):
        #     print(f"& {mean[i]:.1f}\%", end=" ")
        # print("\\\\")
            
    arr_list = []
    ylabels = []
    for relevance, method, model in itertools.product(
        relevances,
        PROMPT_METHODS,
        MODELS,
    ):
        if method != last_method and last_method is not None:
            print_table(arr_list, ylabels)
        if relevance != last_relevance:
            print(f"\n-------------------- {relevance, RELEVANCE_DICT[relevance]} --------------------")
            last_relevance = relevance
        if method != last_method:
            print(f"\n-------------------- {method} --------------------")
            last_method = method
            arr_list.clear()
            ylabels.clear()
    
        matr, key_order = results_dict.get_field_matrix(
            field=metrics,
            fixed_keys={
                "relevance": relevance,
                "method": method,
                "model": model,
                # "dataset": "full",
            }, 
            axis_keys=["dataset"],
            fill_missing=0.0,
        )
        # print(np.array(matr).shape, key_order)

        # print(key_order)
        repr = f"{MODELS_SHORT_DICT[model]}\t"
        arr = np.array(matr).T.flatten()
        if not np.allclose(arr, 0.0, 1e-4):
            arr_list.append(arr)
            ylabels.append(MODELS_SHORT_DICT[model])
    print_table(arr_list, ylabels)

def plot_metrics_line(metric_array, metric):
    # For each prompt method, extract the accuracy values across models and plot them.
    # Create a new figure for the line plot.
    fig, ax = plt.subplots(figsize=(4,3))
    # metric_name = ["BR", "RDR", "AFR"]
    method_name = ["Zero Shot", "CoT", "ICL"]
    # Create the line graph with markers.
    ax.plot(MODELS_SHORT, metric_array[:,0], marker='o', label=method_name[0])
    ax.plot(MODELS_SHORT, metric_array[:,1], marker='o', label=method_name[1])
    ax.plot(MODELS_SHORT, metric_array[:,2], marker='o', label=method_name[2])

    ax.legend()
    ax.set_ylim(0)
    plt.title(metric+" Value")
    plt.tight_layout()
    plt.savefig(os.path.join(stats_folder, "line_plot_"+metric+".pdf"))

import os
import numpy as np
import matplotlib.pyplot as plt

def named_scatter(x, y, keys, title, xlabel, ylabel, scores=None, show=False):
    scatter_plot_path = os.path.join("results/mcq_results/stats", "metric_scatter")
    os.makedirs(scatter_plot_path, exist_ok=True)

    # 1) compute badness = distance from origin
    x_arr = np.array(x, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    badness = np.sqrt(x_arr**2 + y_arr**2)
    normed = (badness - badness.min()) / (badness.max() - badness.min())

    # 2) map through green→red
    cmap = plt.get_cmap('RdYlGn_r')
    colors = cmap(normed)

    # optional: keep a cycle of distinct markers if you still want shape variety
    markers = ['o', 'X', 'd', '*']

    fig, ax = plt.subplots(figsize=(6, 8))

    # hide the top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # make sure scores is a plain Python list if it exists
    if scores is not None:
        scores_list = list(scores)
        # label each model with its score
        labels = [f"{k} ({int(s)})" for k, s in zip(keys, scores_list)]
    else:
        # no scores → just use the key names
        scores_list = [None] * len(keys)
        labels = list(keys)

    for i, (xi, yi, lbl) in enumerate(zip(x_arr, y_arr, labels)):
        ax.scatter(
            xi, yi,
            marker=markers[i % len(markers)],
            s=100,
            color=colors[i],
            edgecolor='k',
            linewidth=1,
            alpha=1,
            label=lbl,
        )

    # only draw the little text labels if we actually have scores
    if scores is not None:
        for i, (xi, yi, s) in enumerate(zip(x_arr, y_arr, scores_list)):
            if i == 9:
                print("Modifying")
                x_offset = xi + 0.15
                y_offset = yi - 0.8
            else:
                x_offset = xi + 0.15
                y_offset = yi + 0.1
            ax.text(
                x_offset, y_offset,
                str(int(s)),
                fontsize=16,
                va='bottom', ha='left'
            )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(which='both', linestyle='--', alpha=0.5)

    # 3) colorbar for the badness scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0,1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label('Badness (distance from origin)', fontsize=14)

    # legend below
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
        fontsize=16,
        markerscale=1.5,
        handletextpad=0.5
    )

    plt.tight_layout(rect=[0,0,1,1])
    outpath = os.path.join(scatter_plot_path, title.replace(" ", "_") + ".pdf")
    plt.savefig(
        outpath, 
        bbox_inches='tight',    # <= include all labels
        pad_inches=0.1          # <= optional extra padding
    )
    if show:
        plt.show()
    else:
        plt.close()



def rel_irrel_backback_barplot(
    rel_results: np.ndarray,
    irrel_results: np.ndarray,
    models: list[str],
    title: str,
    show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Back-to-back horizontal bar plot with model names on the left.
    Bars on the left (red) and right (blue), with color intensity
    mapped from [0,100] → [0.2,0.6].

    rel_results and irrel_results should be arrays of the same length
    with values in [0,100].
    """
    # ensure numpy arrays
    rel = np.array(rel_results, dtype=float)
    irr = np.array(irrel_results, dtype=float)
    assert rel.shape == irr.shape == (len(models),), "Lengths must match models"

    # rel_colors = cm.Blues(rel_norm)
    # irr_colors = cm.Oranges(irr_norm)
    rel_color = cm.Blues([0.66])
    irr_color = cm.Blues([0.33])    # cm.Blues([0.7])

    n = len(models)
    y = np.arange(n)

    # compute padding for annotations and x-limits
    # pad = 1.0  # will be overridden after computing limits
    # max_val = max(rel.max(), irr.max(), 1.0)
    lpad = rel.max() * 0.02
    rpad = irr.max() * 0.02
    left_lim  = rel.max() * (-1.1) - 10
    right_lim =  irr.max()

    # create directories
    rel_irrel_path = os.path.join("results/mcq_results/stats", "rel_irrel_barplot")
    os.makedirs(rel_irrel_path, exist_ok=True)

    # set up figure + GridSpec
    fig = plt.figure(constrained_layout=False, figsize=(6, 3))
    gs = fig.add_gridspec(2, 1,
                          height_ratios=[0.1, 1],
                          hspace=0.2, wspace=0.1)

    # — legend row —
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis('off')
    handles = [patches.Patch(color=c, label=m, hatch=h) for c,m,h in zip(
        [rel_color, irr_color], 
        ["Relevant", "Mixed"],
        ["", "\\\\\\"],
    )]
    for h in handles:
        h._hatch_color = mcolors.to_rgba("navy", alpha=0.5)
        h.stale = True
        h._hatch_linewidth = 0.5
    ax_leg.legend(handles=handles,
                  loc='center',
                  ncol=2,
                  frameon=False,
                  fontsize=18)

    # — the 2×2 bar plots (we’ll drop extras below) —
    ax = fig.add_subplot(gs[1, 0])

    # left (negative) bars
    ax.barh(y, -rel, color=rel_color, height=0.6)
    # right bars
    right_bcs = ax.barh(y, irr, color=irr_color, hatch='\\\\\\', height=0.6)
    for bc in right_bcs:
        bc._hatch_color = mcolors.to_rgba("navy", alpha=0.5)
        bc.stale = True
        bc._hatch_linewidth = 0.5

    # y-axis with model names on the left
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=18)
    ax.invert_yaxis()

    # annotate integer values just outside the bars
    for yi, val in zip(y, rel):
        ax.text(-val-1, yi, f"{int(val)}", ha='right', va='center', fontsize=18)
    for yi, val in zip(y, irr):
        ax.text(val+1, yi, f"{int(val)}", ha='left', va='center', fontsize=18)

    # hide spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    # fig.suptitle(title, fontsize=15)
    ax.set_xlim(left_lim, right_lim)
    # ax.set_xlabel("Relevant Preferences    |    Mixed Preferences   ", fontsize=12)
    plt.tight_layout()
    # save
    fname = title.replace(" ", "_") + ".pdf"
    plt.savefig(
        os.path.join(rel_irrel_path, fname),
        bbox_inches='tight',    # <= include all labels
        pad_inches=0.1          # <= optional extra padding
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def metric_heatmap(matrix, title, xticks, yticks, xlabel, ylabel, show=False):
    heatmap_path = os.path.join("results/mcq_results/stats", "metric_heatmap")
    os.makedirs(heatmap_path, exist_ok=True)
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 8))

    # Create a heatmap with annotations for each block showing its value
    # Create a heatmap with annotations for each block showing its value
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, square=True)
    
    # Rotate the x-axis labels (model names) by 45 degrees.
    ax.set_xticklabels(xticks, rotation=45, ha='right')
    ax.set_yticklabels(yticks, rotation=0)

    # Add titles and axis labels for clarity
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_path, title.replace(" ", "_") + ".pdf"))
    if show:
        plt.show()
    else:
        plt.close()


def relevance_hbarplot(
    data: np.ndarray,
    relevance_list: list[str],
    models: list[str],
    title, 
    xlabel,
    label_dict=RELEVANCE_DICT,
    figh=5,
    show: bool = False
):
    """
    data.shape == (n_models, n_rel)
    draws one grouped horizontal‐bar plot with a custom legend row above
    """

    # output folder
    hbar_path = os.path.join("results/mcq_results/stats", "relevance_hbarplot")
    os.makedirs(hbar_path, exist_ok=True)

    n_models = len(models)
    n_rel    = len(relevance_list)

    # positions for each model
    y = np.arange(n_models)
    bar_h = 0.8 / n_rel

    # colors & map
    cmap      = cm.Blues
    colors    = cmap(np.linspace(0.2, 1.0, n_rel))
    color_map = dict(zip(relevance_list, colors))

    # ─── set up figure + GridSpec ─────────────────────────────────────────────
    fig = plt.figure(constrained_layout=False, figsize=(6, figh))
    gs  = fig.add_gridspec(2, 1,
                           height_ratios=[0.1, 0.9],
                           hspace=0.02)

    # — legend row —
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.axis('off')
    handles = [
        patches.Patch(color=color_map[r], label=label_dict[r])
        for r in relevance_list
    ]
    ax_leg.legend(handles=handles,
                  loc='center',
                  ncol=n_rel,
                  frameon=False,
                  fontsize=18)

    # — bar plot row —
    ax = fig.add_subplot(gs[1, 0])

    for i, rel in enumerate(relevance_list):
        y_pos = y + (i - (n_rel - 1) / 2) * bar_h
        ax.barh(
            y_pos,
            data[:, i],
            height=bar_h,
            color=color_map[rel],
            edgecolor='none'
        )

    # style the bar plot
    ax.axvline(0, color='gray', linestyle='--', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=18)
    # ax.set_title(title, fontsize=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(which='both', axis='x', linestyle='--', alpha=0.3)

    # ─── save & optionally show ────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(
        os.path.join(hbar_path, f"{title.replace(' ', '_')}.pdf"),
        bbox_inches='tight',
        pad_inches=0.1
    )
    if show:
        plt.show()
    else:
        plt.close()

def rel_irrel_topdown_barplot(rel_results, irrel_results, models, title, show=False):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as patches

    # prep data + colors
    rel = np.asarray(rel_results, float)
    irr = np.asarray(irrel_results, float)
    n = len(models)
    x = np.arange(n)
    bar_w = 0.6 / 2  # two bars per model
    rel_color = cm.Blues([0.66])[0]
    irr_color = cm.Blues([0.33])[0]

    # output dir
    out = os.path.join("results/mcq_results/stats", "rel_irrel_barplot")
    os.makedirs(out, exist_ok=True)

    # figure
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    # zero line at top
    ax.axhline(0, color="gray", linestyle="-", lw=1)

    # draw each pair of bars downward (negative height)
    for i, (rv, iv) in enumerate(zip(rel, irr)):
        ax.bar(x[i] - bar_w/2, rv, bar_w,
               color=rel_color, edgecolor="none")
        ax.bar(x[i] + bar_w/2, iv, bar_w,
               facecolor=irr_color, edgecolor="navy", hatch="///", linewidth=0.0)

        # annotations just below the bar tips
        pad = max(rel.max(), irr.max()) * 0.02
        ax.text(x[i] - bar_w/2, rv - pad,
                f"{int(rv)}", ha="center", va="top")
        ax.text(x[i] + bar_w/2, iv - pad,
                f"{int(iv)}", ha="center", va="top")

    # x-axis labels at bottom
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14, rotation=45, ha="right")

    # remove spines/ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])

    # flip the y-axis so negative (downward) grows downward
    ax.invert_yaxis()

    # legend
    handles = [
        patches.Patch(color=rel_color, label="Relevant"),
        patches.Patch(facecolor=irr_color, edgecolor="navy", hatch="///", label="Mixed")
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=14)

    # title + save
    ax.set_title(title, fontsize=16, pad=12)
    fname = title.replace(" ", "_") + ".pdf"
    plt.savefig(os.path.join(out, fname), bbox_inches="tight", pad_inches=0.1)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def method_hbarplot(matrices, methods, models, subplot_titles, title, show=False):
    """
    Matrix: (models x methods)
    """
    hbar_path = os.path.join("results/mcq_results/stats", "method_hbarplot")
    os.makedirs(hbar_path, exist_ok=True)

    n_models  = len(models)
    n_methods = len(methods)
    n_metrics = len(matrices)
    y = np.arange(n_models)
    bar_h = 0.8 / n_methods

    # color mapping
    cmap = cm.Blues
    colors = cmap(np.linspace(0.2, 1.0, n_methods))
    color_map = dict(zip(methods, colors))

    # set up figure + GridSpec
    fig = plt.figure(constrained_layout=False, figsize=(12, 4))
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[0.1, 1],
                          hspace=0.2, wspace=0.2)

    # — legend row —
    ax_leg = fig.add_subplot(gs[0, :])
    ax_leg.axis('off')
    handles = [patches.Patch(color=color_map[m], label=m) for m in methods]
    ax_leg.legend(handles=handles,
                  loc='center',
                  ncol=n_methods,
                  frameon=False,
                  fontsize=15)

    # — the 2×2 bar plots (we’ll drop extras below) —
    axs = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1,2]),
    ]

    # plot only as many as we have matrices
    for axn, (ax, mat, sub_title) in enumerate(zip(axs, matrices, subplot_titles)):
        for i, m in enumerate(methods):
            off = (i - (n_methods - 1) / 2) * bar_h
            ax.barh(
                y + off,
                mat[:, i],
                height=bar_h,
                color=color_map[m],
                edgecolor='none'
            )
        ax.axvline(0, color='gray', linestyle='--', lw=0.8)
        if axn == 0:
            print("Setting y-ticks")
            ax.set_yticks(y)
            ax.set_yticklabels(models, fontsize=15)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        ax.set_xlabel(METRIC_LABELS[sub_title], fontsize=15)
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

    # remove any unused axes (e.g. the 4th when n_metrics == 3)
    for ax in axs[n_metrics:]:
        fig.delaxes(ax)

    # # if odd number of metrics, center the last one in its row
    # if n_metrics % 2 == 1:
    #     last_ax = axs[n_metrics - 1]
    #     ax_box = last_ax.get_position()
    #     ax_width = ax_box.width
    #     new_left = 0.5 - (ax_width / 2)
    #     last_ax.set_position([new_left, ax_box.y0, ax_width, ax_box.height])

    # optional super-title
    # fig.suptitle(title, fontsize=16, y=0.95)

    # save/show
    plt.savefig(os.path.join(hbar_path, title.replace(" ", "_") + ".pdf"),
                format='pdf', bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    else:
        plt.close()


def missing_answer_diffplot(data, keys, title, xlabel, series_labels=None, show=False):
    """
    Plot (high - low) for each key, sorted by that difference.
    Uses a viridis colormap and shows a zero line for negative/positive.

    keys:  length-N list of labels
    """
    # prepare output directory
    hbar_path = os.path.join("results/mcq_results/stats", "missing_ans_diffplot")
    os.makedirs(hbar_path, exist_ok=True)

    # unpack & compute difference
    data = np.asarray(data)
    M, N = data.shape

    # sort by diff
    # order = np.argsort(diff)
    order = np.argsort(data.mean(axis=0))
    data   = data[:, order]
    labels = [MODELS_SHORT_DICT[keys[i]] for i in order]

    # normalize data for colormap
    # cmap = cm.plasma
    # norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
    # colors = cmap(norm(data))
    colors = cm.Blues([0.45])

    # 2) Prepare series labels & colors
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in range(M)]

    # 3) Bar geometry
    y = np.arange(N)
    total_height = 0.6
    bar_h = total_height / M
    # offsets so bars are centered around each y
    offsets = np.linspace(
        -total_height/2 + bar_h/2,
         total_height/2 - bar_h/2,
         M
    )

    # create figure & axis
    #fig, ax = plt.subplots(figsize=(6, 6))

    # horizontal bars
    # ax.barh(y, data, height=0.8, color=colors, edgecolor='none')
    fig, ax = plt.subplots(figsize=(6 + M, max(6, N * 0.4)))
    for m in range(M):
        if series_labels is None:
            ax.barh(
                y + offsets[m],
                data[m],
                height=bar_h,
                color=colors[m],
                edgecolor='none'
            )
        else:
            ax.barh(
                y + offsets[m],
                data[m],
                height=bar_h,
                color=colors[m],
                label=series_labels[m],
                edgecolor='none'
            )

    for i, yi in enumerate(y):
        for field in range(data.shape[0]):
            val = data[field, i]
            if val <= 0:
                ax.text(1, yi, f"{int(val)}", ha='left', va='center', fontsize=18)
            else:
                ax.text(val+1, yi, f"{int(val)}", ha='left', va='center', fontsize=18)
            # ax.text(val+1, yi, f"{int(val)}", ha='left', va='center', fontsize=18)
        
    # zero reference line
    ax.axvline(0, color='gray', linestyle='--', lw=0.8)

    # labels & styling
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=18)
    # ax.set_title(title)

    # grid behind bars
    ax.set_axisbelow(True)
    ax.grid(axis='x', which='major',
            linestyle='--', linewidth=0.5,
            color='gray', alpha=0.3)

    # remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    # add a horizontal colorbar for the data scale
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax,
    #                     orientation='horizontal',
    #                     pad=0.15,
    #                     fraction=0.05)
    # cbar.set_label('Low - High (Percentage)', fontsize=12)

    plt.tight_layout()

    # save or show
    out_file = os.path.join(hbar_path, title.replace(" ", "_") + ".pdf")
    plt.savefig(out_file, format='pdf', bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    else:
        plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

def plot_paired_violin(
    data_left,
    data_right,
    labels,
    title=None,
    kind='violin',         # 'violin' or 'gaussian'
    bw_method=0.3,
    half_width=0.4,
    alpha=1.0,
    show=False,
):
    """
    Draw back-to-back half-violins.

    data_left, data_right : list of 1D arrays, length N
    labels                : list of N category names
    """
    # prepare output directory
    violin_path = os.path.join("results/mcq_results/stats", "full_sample_acc_violin")
    os.makedirs(violin_path, exist_ok=True)

    N = len(labels)
    positions = np.arange(N)

    fig, ax = plt.subplots(figsize=(8, 6))

    color_left = cm.Blues([0.66])
    color_right = cm.Oranges([0.5])

    # find common y-limits so all violins share the same vertical span
    all_data = np.concatenate(data_left + data_right)
    y_min, y_max = all_data.min(), all_data.max()
    y_vals = np.linspace(y_min, y_max, 200)

    for i, (dl, dr) in enumerate(zip(data_left, data_right)):
        x0 = positions[i]

        if kind == 'violin':
            # KDE for left
            kde_l = gaussian_kde(dl, bw_method=bw_method)
            dens_l = kde_l(y_vals)
            dens_l = dens_l / dens_l.max() * half_width
            ax.fill_betweenx(
                y_vals,
                x0, x0 - dens_l,
                facecolor=color_left, alpha=alpha, linewidth=0
            )
            # KDE for right
            kde_r = gaussian_kde(dr, bw_method=bw_method)
            dens_r = kde_r(y_vals)
            dens_r = dens_r / dens_r.max() * half_width
            ax.fill_betweenx(
                y_vals,
                x0, x0 + dens_r,
                facecolor=color_right, alpha=alpha, linewidth=0
            )

        elif kind == 'gaussian':
            # Fit Normal to left
            mu_l, sigma_l = np.mean(dl), np.std(dl)
            pdf_l = norm.pdf(y_vals, loc=mu_l, scale=sigma_l)
            dens_l = pdf_l / pdf_l.max() * half_width
            ax.fill_betweenx(
                y_vals,
                x0, x0 - dens_l,
                facecolor=color_left, alpha=alpha, linewidth=0
            )
            # Fit Normal to right
            mu_r, sigma_r = np.mean(dr), np.std(dr)
            pdf_r = norm.pdf(y_vals, loc=mu_r, scale=sigma_r)
            dens_r = pdf_r / pdf_r.max() * half_width
            ax.fill_betweenx(
                y_vals,
                x0, x0 + dens_r,
                facecolor=color_right, alpha=alpha, linewidth=0
            )

        else:
            raise ValueError(f"Unknown kind: {kind!r}. Use 'violin' or 'gaussian'.")


    # styling
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=24)
    ax.set_xlim(-1, N)  # give a bit of breathing room
    ax.set_ylabel('Accuracy', fontsize=24)
    # ax.set_title(title, fontsize=15)

    # legend
    left_patch  = plt.Line2D([0], [0], color=color_left,  lw=10, alpha=alpha)
    right_patch = plt.Line2D([0], [0], color=color_right, lw=10, alpha=alpha)
    ax.legend(
        [left_patch, right_patch],
        ['Within', 'Between'],
        loc='upper right',
        frameon=False,
        fontsize=18,
    )

    ax.grid(axis='y', color='lightgrey', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    # save or show
    out_file = os.path.join(violin_path, title.replace(" ", "_") + ".pdf")
    plt.savefig(out_file, format='pdf', bbox_inches='tight', dpi=600)
    if show:
        plt.show()
    else:
        plt.close()


def compare_profile_accuracy(
    model_name,
    indiv_dataset_paths,
    results_dict,
    show=False,
):
    acc_full = []
    acc_sampled = []
    assert len(indiv_dataset_paths) == len(DATASETS), "Dataset paths must match datasets"
    for dataset, fpath in zip(DATASETS, indiv_dataset_paths):
        path = os.path.join("results/mcq_results/relevant/direct", dataset, fpath)
        if not os.path.exists(path):
            assert False, f"Warning: No file found for {dataset} at {path}"
        df = pd.read_csv(path)
        # all_pref_dfs[dataset] = df

        # Compute full accuracy
        pref_acc_dict_full = {}
        i = 1
        soln = df["gold_option"]
        while f"profile_{i}_answer" in df.columns:
            prof_answer = df[f"profile_{i}_answer"]
            # Compute the accuracy
            accuracy = (soln == prof_answer).mean()
            pref_acc_dict_full[i] = accuracy
            i += 1

        # Get partial
        partial_df = results_dict.get(
            field="df",
            relevance="relevant",
            method="direct",
            model=model_name,
            dataset=dataset,
        )
        assert len(partial_df) == 1
        partial_df = list(partial_df.values())[0]
        
        pref_acc_dict_sampled = {}
        # Retrieve chunks for each profile
        for i in pref_acc_dict_full.keys():
            profile_set = DATASET_PREFERENCES[dataset]
            profile = profile_set[i]
            profile_chunk = partial_df[partial_df["preference"] == profile]
            # Get the profile answer
            accuracy = profile_chunk["pref_correct"].mean()
            pref_acc_dict_sampled[i] = accuracy

        # Aggregate results
        full_arr = np.array([val for val in pref_acc_dict_full.values()])
        acc_full.append(full_arr)
        partial_arr = np.array([val for val in pref_acc_dict_sampled.values()])
        acc_sampled.append(partial_arr)

    plot_paired_violin(
        acc_full,
        acc_sampled,
        [DATASET_SHORT_DICT[n] for n in DATASETS], 
        title=f"Profile Accuracy for {model_name}",
        # xlabel="Dataset",
        show=show
    )




def aligner_improvement_hbarplot_full(
    base_results: np.ndarray,
    aligner_results: np.ndarray,
    models: list[str],
    metrics: list[str],
    title: str,
    show: bool = False
) -> tuple[plt.Figure, plt.Axes]:
    """
    Back-to-back grouped horizontal bar plot.
    
    - base_results, aligner_results: arrays of shape (n_models, n_metrics),
      values in [0, 100].
    - models: list of model names, length n_models.
    - metrics: list of metric names, length n_metrics.
    """
    # --- sanity checks ---
    base = np.array(base_results, dtype=float)
    algn = np.array(aligner_results, dtype=float)
    n_models, n_metrics = base.shape
    assert algn.shape == (n_models, n_metrics)
    assert len(models) == n_models
    assert len(metrics) == n_metrics

    # --- colors ---
    blues  = cm.Blues(np.linspace(0.3, 0.7, n_metrics))
    oranges = cm.Oranges(np.linspace(0.3, 0.7, n_metrics))


    # ─── set up figure + GridSpec ─────────────────────────────────────────────
    fig = plt.figure(constrained_layout=False, figsize=(5, 8))
    gs  = fig.add_gridspec(2, 1,
                           height_ratios=[0.05, 0.95],
                           hspace=0.02)

    n_metrics = len(metrics)
    # — legend row —
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.axis('off')
    handles = [
        patches.Patch(color=blues[0], alpha=0.0, label=f"Zero-Shot:"), 
        patches.Patch(color=oranges[0], alpha=0.0, label=f"Aligner:")
    ]
    for i in range(n_metrics):
        handles.extend([
            patches.Patch(color=blues[i], label=f"{metrics[i]}"), 
            patches.Patch(color=oranges[i], label=f"{metrics[i]}")
        ])
    ax_leg.legend(handles=handles,
                  loc='center',
                  ncol=len(metrics)+1,
                  frameon=False,
                  fontsize=12)

    # --- bar plot ---
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(n_models)

    # compute offsets so metrics are grouped around each y tick
    bar_height = 0.8 / n_metrics
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_height

    # global padding & limits
    max_val = max(base.max(), algn.max(), 1.0)
    pad = max_val * 0.01
    left_lim = -base.max() * 1.5 - 2
    right_lim = algn.max()

    # draw bars for each metric
    for j in range(n_metrics):
        yj = y + offsets[j]
        # baseline (negative)
        ax.barh(
            yj, -base[:, j],
            height=bar_height,
            color=blues[j],
        )
        # aligner (positive)
        ax.barh(
            yj, algn[:, j],
            height=bar_height,
            color=oranges[j],
        )
        # annotate values
        for yi, bv, av in zip(yj, base[:, j], algn[:, j]):
            ax.text(-bv - pad, yi, f"{int(bv)}",
                    ha='right', va='center', fontsize=14)
            ax.text(av + pad, yi, f"{int(av)}",
                    ha='left',  va='center', fontsize=14)

    # y-axis labels
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.invert_yaxis()

    # styling
    ax.set_xlim(left_lim, right_lim)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # title
    # fig.suptitle(title, fontsize=14, y=0.98)

    # save
    rel_irrel_path = os.path.join(
        "results/mcq_results/stats",
        "aligner_improv_hbarplot"
    )
    os.makedirs(rel_irrel_path, exist_ok=True)
    fname = title.replace(" ", "_") + ".pdf"
    plt.savefig(
        os.path.join(rel_irrel_path, fname),
        bbox_inches='tight',
        pad_inches=0.1
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax



def aligner_improvement_hbarplot(
    data: np.ndarray,
    methods: list[str],
    models: list[str],
    title: str,
    xlabel: str,
    label_dict,       # e.g. RELEVANCE_DICT
    figh: float = 4,
    show: bool = False
):
    """
    data.shape == (n_models, n_methods)
    draws one grouped horizontal‐bar plot with a custom legend row above,
    using two Blues (0.66, 0.33) and a '*' hatch on the lighter one.
    """

    # output folder
    hbar_path = os.path.join("results/mcq_results/stats", "aligner_improv_hbarplot")
    os.makedirs(hbar_path, exist_ok=True)

    n_models = len(models)
    n_meth   = len(methods)

    # bar geometry
    y     = np.arange(n_models)
    bar_h = 0.6 / n_meth

    # colors & hatches
    dark_blue  = cm.Blues(0.66)
    light_blue = cm.Greens(0.4)
    colors     = [dark_blue, light_blue]
    hatch_map  = {methods[0]: '', methods[1]: ''}

    color_map = dict(zip(methods, colors))

    # ─── set up figure + GridSpec ─────────────────────────────────────────────
    fig = plt.figure(constrained_layout=False, figsize=(6, figh))
    gs  = fig.add_gridspec(2, 1,
                           height_ratios=[0.1, 0.9],
                           hspace=0.02)

    # — legend row —
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.axis('off')

    handles = []
    for m in methods:
        p = patches.Patch(color=color_map[m],
                          label=label_dict[m],
                          hatch=hatch_map[m])
        # style the hatch
        p._hatch_color      = mcolors.to_rgba("lime", alpha=0.5)
        p._hatch_linewidth  = 0.5
        p.stale             = True
        handles.append(p)

    ax_leg.legend(handles=handles,
                  loc='center',
                  ncol=n_meth,
                  frameon=False,
                  fontsize=18)

    # — bar plot row —
    ax = fig.add_subplot(gs[1, 0])
    for i, m in enumerate(methods):
        y_pos = y + (i - (n_meth - 1) / 2) * bar_h
        bar = ax.barh(
            y_pos,
            data[:, i],
            height=bar_h,
            color=color_map[m],
            hatch=hatch_map[m],
            edgecolor='none',
        )
        # if this series has a hatch, style it
        if hatch_map[m]:
            for bc in bar:
                bc._hatch_color     = mcolors.to_rgba("", alpha=0.5)
                bc._hatch_linewidth = 0.5
                bc.stale            = True

    # style the bar plot
    ax.axvline(0, color='gray', linestyle='--', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=18)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(which='both', axis='x', linestyle='--', alpha=0.3)

    # ─── save & optionally show ────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(
        os.path.join(hbar_path, f"{title.replace(' ', '_')}.pdf"),
        bbox_inches='tight',
        pad_inches=0.1
    )
    if show:
        plt.show()
    else:
        plt.close()