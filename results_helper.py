import os
import glob
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# -----------------------------
# Global variables
# -----------------------------

RELEVANCE = ["relevant", "irrelevant_set"]
# Prompt methods to process
PROMPT_METHODS = ["direct", "cot", "icl"]

# List of models (as provided)
MODELS_FULL = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "kaist-ai/janus-7b",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "openai/gpt-4o-mini-2024-07-18",
    "deepseek/DeepSeek-R1-Distill-Llama-70B-free"
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

# The input dataset directories (each should contain a CSV per model)
DATASETS_FULL = [
    "tau/commonsense_qa",
    "cais/mmlu",
    "truthfulqa/truthful_qa",
]
# Process dataset names
DATASETS = [dataset.split("/")[-1] for dataset in DATASETS_FULL]

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
        # Read the full dataset
        # if full_path is None or not os.path.exists(full_path):
        #     if verbose:
        #         print(f"Warning: No full dataset found for [{relevance}, {method}, {model}].")
        #     full_df = None
        # else:
        #     full_df = pd.read_csv(full_path)
        #     # Compute extra columns if neded
        #     full_df = compute_full(full_df)
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

def BR(chunk):
    correct_no_pref_chunk = chunk.loc[chunk['nopref_correct'] == 1, :]
    robust_col = correct_no_pref_chunk.loc[:, "is_robust"].mean()
    br = 1 - robust_col
    return br

def RDR(chunk):
    robust_col = chunk.loc[:, "is_robust"].mean()
    corr_no_pref = chunk.loc[:, "nopref_correct"].mean()
    corr_pref = chunk.loc[:, "pref_correct"].mean()
    rdr = 1 - robust_col/max(corr_no_pref, corr_pref)
    return rdr

def AFR(chunk):
    robust_col = chunk.loc[:, "is_robust"].mean()
    corr_pref = chunk.loc[:, "pref_correct"].mean()
    afr = 1 - robust_col/corr_pref
    return afr

def PVR(chunk):
    robust = chunk["is_robust"].astype(bool)
    no_pref = chunk["nopref_correct"].astype(bool)
    # Calculate IoU
    intersection = (robust & no_pref).sum()
    union = (robust | no_pref).sum()
    # Stability
    if union == 0:
        union = 1.0
    pvr = intersection / union
    return pvr


def compute_metrics(results_dict, verbose=False):
    for k, df in results_dict.items(
        field="df",
    ):
        (relevance, method, model, dataset, _) = k
        for name, func in zip(
            ["BR", "RDR", "AFR", "PVR"],
            [BR, RDR, AFR, PVR],
        ):
            percentage = func(df) * 100. if df is not None else None
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

#----------------------------------
# 
# Plot results helper
# 
#----------------------------------

def print_metric_table(results_dict, relevances=RELEVANCE, metrics=["BR", "RDR", "AFR", "PVR"]):
    # Compute metrics
    for model in MODELS:
        # print(f"model}-----------")
        for relevance, method in itertools.product(
            relevances,
            PROMPT_METHODS,
        ):  
            print(f"----------- {model} {relevance} {method} -----------")
            matr, key_order = results_dict.get_field_matrix(
                field=metrics,
                fixed_keys={
                    "relevance": relevance,
                    "method": method,
                    "model": model,
                }, 
                axis_keys=["dataset"],
                fill_missing=0.0,
            )
            # print(key_order)
            print(model, end=" ")
            for j in range(len(matr[0])):
                for i in range(len(matr)):
                    print(f"& {matr[i][j]:.1f}\%", end=" ")
            print()
            # print(np.array(matr).flatten())

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