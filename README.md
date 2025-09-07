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

$Robust(x, P, y) = \begin{cases} Acc(y) \land Followed(y, P) & \text{if; } PrefRel(x, P) = 1 \\ textAcc(y) & \text{if; } PrefRel(x, P) = 0 \text{; or; } P = \emptyset \end{cases}$

![equation](https://latex.codecogs.com/png.latex?\text{Robust}(x,P,y)=\begin{cases}\text{Acc}(y)\land\text{Followed}(y,P)&\text{if}\;\text{PrefRel}(x,P)=1\\\text{Acc}(y)&\text{if}\;\text{PrefRel}(x,P)=0\;\text{or}\;P=\emptyset\end{cases})

## Evaluation

### Dataset
We show our dataset curation pipeline below (see paper for more details)

![Data Curation](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_data_pipeline.png?raw=true)

Our version is available to download @ data/robuset_main.csv

### Metrics
We introduce four complementary error-based metrics. Lower values (closer to zero) across all metrics indicate more robust, stable, and consistent behavior. 




## Research Questions/Results
Some of the key research questions and results are summarized below:

RQ1

**Q: Are LLMs robust when we include a relevant user preference?**

**Answer: No.** We evaluate several models across the metrics listed above and we find that most models suffer some form of degradation in robustness. 

RQ4
**Q: How robust are LLMs when both relevant and irrelevant preferences are present?**

**Answer: Irrelevant preferences amplify robustness errors.** Here, we evaluate LLM Robustness on a preference list (like in real world) with an irrelevant preference setting and mixed (relevant and irrelevant) preference setting. Our results show that the presence of irrelevant preferences amplifies alignment errors (ie, LLMs struggle to delineate between the relevant and irrelevant preferences within the preference list). This is evident in the substantial increase in alignment failure, leading to an increase in robustness error across all models when compared to the single relevant preference setting.

RQ5
**Q: What types of failure patterns do models exhibit?**

**Answer:  Question and preference categories significantly influence robustness.** For questions drawn from TruthfulQA, which are often short and straightforward, preferences eliciting clarity and conciseness have the least breakage rate, and preferences that require contextual details or practical examples have a higher breakage rate. We conjecture that this is because context/thinking related preferences make models overthink, which leads to incorrect answers. For MMLU, we do not observe any consistent pattern, likely due to its coverage of diverse academic domains. This highlights the complexities and comprehensive scenarios covered in PERG.

## Improving Robustness: Pref-Aligner

We introduce Pref-Aligner, a two-stage agentic framework, which decouples generation from personalization with an agent specialized for each task. In the first stage, a generation agent responds to user queries without considering their defined preferences (if any). In the second stage, the aligner agent takes the unconditioned response from the generation agent, the user preference(s), and produces an aligned response (if needed). That way, we eliminate the inconsistencies resulting from preference signals during initial generation. 

Results show that our framework consistently improves robustness across the representative models we evaluated -   Llama3-8B, Llama3-70B, Mistral-8x7b, and Gemma-9B models. 
Notably, the breakage rate for *Llama-70B* drops from 5.6% to 1.3% in relevant preference settings and remain consistent even in mixed and irrelevant preference settings, highlighting the effectiveness of our proposed framework in diverse conditions.

## Conclusion


