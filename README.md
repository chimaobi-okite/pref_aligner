# Benchmarking and Improving LLM Robustness for Personalized Generation

## Motivation

Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user’s preferences, we argue that factuality is an equally important yet often overlooked dimension.

| ![Figure 1a](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_math_example.png?raw=true) | ![Figure 1b](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_vege_example.png?raw=true) |
|:---:|:---:|
| **Figure 1a:** The example is from Mistral 7B<sub>Instruct</sub>. When prompted with certain preferences, the model's response aligns with the user preference, but fails the question as the preference affects the model's reasoning. | **Figure 1b:** The model hallucinates a non-existent restaurant to match the user’s preference. Without jointly evaluating factuality and preference alignment we risk overestimating model capabilities. |


## Problem Formulation
Let **x** be a user query, **P = {p₁, …, pₙ}** the set of user features/preference set, and **M** a language model.  
Given input (x, P), the model outputs:

![eq1](https://latex.codecogs.com/png.latex?y%20=%20M(x,%20P))

We define the following binary functions:

- ![eq2](https://latex.codecogs.com/png.latex?%5Ctext%7BAcc%7D(y)%20=%201) if y is factually correct w.r.t. x; else 0.  
- ![eq3](https://latex.codecogs.com/png.latex?%5Ctext%7BPrefRel%7D(x,%20P)%20=%201) if some pₓ ∈ P is relevant to x; else 0.  
- ![eq4](https://latex.codecogs.com/png.latex?%5Ctext%7BFollowed%7D(y,%20P)%20=%201) if y incorporates a relevant pₓ ∈ P; else 0.


We say a model \(M\) is said to be *robust* iff:  

**(1)** It maintains factual accuracy while conditioning on the relevant pᵢ ∈ P for any given query x.  
**(2)** It ignores irrelevant user features within the feature set P for any given query x.

![equation](https://latex.codecogs.com/png.latex?\text{Robust}(x,P,y)=\begin{cases}\text{Acc}(y)\land\text{Followed}(y,P)&\text{if}\;\text{PrefRel}(x,P)=1\\\text{Acc}(y)&\text{if}\;\text{PrefRel}(x,P)=0\;\text{or}\;P=\emptyset\end{cases})

## Evaluation

### Dataset
We show our dataset curation pipeline below (see paper for more details)

![Data Curation](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_data_pipeline.png?raw=true)

Our version is available to download @ data/robuset_main.csv

### Metrics

## Research Questions/Results
RQ1
RQ4
RQ5

## Improving Robustness: Pref-Aligner
little description and image of framework and result

## Conclusion


