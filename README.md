# Benchmarking and Improving LLM Robustness for Personalized Generation

## Motivation

Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user’s preferences, we argue that factuality is an equally important yet often overlooked dimension.

| ![Figure 1a](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_math_example.png?raw=true) | ![Figure 1b](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_vege_example.png?raw=true) |
|:---:|:---:|
| **Figure 1a:** The example is from Mistral 7B<sub>Instruct</sub>. When prompted with certain preferences, the model's response aligns with the user preference, but fails the question as the preference affects the model's reasoning. | **Figure 1b:** The model hallucinates a non-existent restaurant to match the user’s preference. Without jointly evaluating factuality and preference alignment we risk overestimating model capabilities. |


## Problem Formulation
Let **x** be a user query, **P = {p₁, …, pₙ}** the set of user features/preference set, and **M** a language model.  
Given input (x, P), the model outputs:

* $y = M(x, P)$

The following are the binary functions:

* $\text{Acc}(y) = 1$ if y is factually correct w.r.t. x; else 0.
* $\text{PrefRel}(x, P) = 1$ if some $p_x \in P$ is relevant to x; else 0.
* $\text{Followed}(y, P) = 1$ if y incorporates a relevant $p_x \in P$; else 0.

We say a model \(M\) is said to be *robust* iff:  

**(1)** It maintains factual accuracy while conditioning on the relevant pᵢ ∈ P for any given query x.  
**(2)** It ignores irrelevant user features within the feature set P for any given query x.

$$\text{Robust}(x,P,y)=\begin{cases}\text{Acc}(y)\land\text{Followed}(y,P)&\text{if}\;\text{PrefRel}(x,P)=1\\&\text{Acc}(y)&\text{if}\;\text{PrefRel}(x,P)=0\;\text{or}\;P=\emptyset\end{cases}$$

<!-- ![equation](https://latex.codecogs.com/png.latex?\text{Robust}(x,P,y)=\begin{cases}\text{Acc}(y)\land\text{Followed}(y,P)&\text{if}\;\text{PrefRel}(x,P)=1\\\text{Acc}(y)&\text{if}\;\text{PrefRel}(x,P)=0\;\text{or}\;P=\emptyset\end{cases}) -->

## Evaluation

### Dataset
We show our dataset curation pipeline below (see paper for more details)

![Data Curation](https://github.com/chimaobi-okite/pref_aligner/blob/main/paper_images/jpg_data_pipeline.png?raw=true)

Our version is available to download @ data/robuset_main.csv

### Metrics
We introduce four complementary error-based metrics. Lower values (closer to zero) across all metrics indicate more robust, stable, and consistent behavior. 

**Breakage Rate**

Measures how often personalization causes the model to fail on inputs that it handles correctly without any preference conditioning.

Formally,

$$\text{Breakage Rate} = 1 - \mathbb{E}_{x \in Q^*}[\text{Acc}_{\text{pref}}(y)]$$
<!-- Given $Q$ is all query set in our dataset $D$, then $Q^* = \{x \in Q \mid \text{Acc}_{\text{no-pref}}(y) = 1\}$, $\text{Acc}_{\text{pref}}(y)$ and $\text{Acc}_{\text{no-pref}}(y)$ are the accuracy of generating $y$ with and without any preference, respectively. -->

**Alignment Failure**

Measures among examples where the model answers correctly without personalization, how often the model fails to align with user preferences.

$$\text{Alignment Failure} = 1 - \mathbb{E}_{x \in Q^*}[\text{Followed}(y, P)].$$


**Robustness Error**

Robustness Error is the union of breakage and alignment failure sets and measures how often the model either fails to answer it correctly or aligns with user preference. Formally:

$$
\text{Robustness Error} = 1 - \mathbb{E}_{x \in Q^*} \left[
\text{Acc}_{\text{pref}}(y) \ \cap\ \text{Followed}(y, P)
\right] \\
= 1 - \mathbb{E}_{x \in Q^*} \left[\text{Robust}(x, P, y)\right]
$$

**Performance Variation**

Measures the divergence in correctness with and without personalization.

Similar to Jaccard distance, we define it as:

$$\text{Performance Variation} = 1 - \frac{|\mathcal{A}_{\text{pref}} \cap \mathcal{A}_{\text{no-pref}}|}{|\mathcal{A}_{\text{pref}} \cup \mathcal{A}_{\text{no-pref}}|},$$
<!-- where $\mathcal{A}_{\text{pref}}$ and $\mathcal{A}_{\text{no-pref}}$ denote the sets of correctly answered questions with and without preference conditioning, respectively. -->

## Research Questions/Results
Some of the key research questions and results are summarized below:

**Q: Are LLMs robust when we include a relevant user preference?**

**Answer: No.** We evaluate several models across the metrics listed above and we find that most models suffer some form of degradation in robustness. Models exhibit varying levels of breakage and alignment failures, that can lead to a combined robustness as bad as upto 34% in some of the less robust models, and even as high as 9% in some of the more robust models. 

**Q: How robust are LLMs when both relevant and irrelevant preferences are present?**

**Answer: Irrelevant preferences amplify robustness errors.** Here, we evaluate LLM Robustness on a preference list (like in real world) with an irrelevant preference setting and mixed (relevant and irrelevant) preference setting. Our results show that the presence of irrelevant preferences amplifies alignment errors (ie, LLMs struggle to delineate between the relevant and irrelevant preferences within the preference list). This is evident in the substantial increase in alignment failure, leading to an increase in robustness error across all models when compared to the single relevant preference setting.

**Q: What types of failure patterns do models exhibit?**

**Answer:  Question and preference categories significantly influence robustness.** For questions drawn from TruthfulQA, which are often short and straightforward, preferences eliciting clarity and conciseness have the least breakage rate, and preferences that require contextual details or practical examples have a higher breakage rate. We conjecture that this is because context/thinking related preferences make models overthink, which leads to incorrect answers. For MMLU, we do not observe any consistent pattern, likely due to its coverage of diverse academic domains. This highlights the complexities and comprehensive scenarios covered in PERG.

## Improving Robustness: Pref-Aligner

We introduce Pref-Aligner, a two-stage agentic framework, which decouples generation from personalization with an agent specialized for each task. In the first stage, a generation agent responds to user queries without considering their defined preferences (if any). In the second stage, the aligner agent takes the unconditioned response from the generation agent, the user preference(s), and produces an aligned response (if needed). That way, we eliminate the inconsistencies resulting from preference signals during initial generation. 

Results show that our framework consistently improves robustness across the representative models we evaluated -   Llama3-8B, Llama3-70B, Mistral-8x7b, and Gemma-9B models. 
Notably, the breakage rate for *Llama-70B* drops from 5.6% to 1.3% in relevant preference settings and remain consistent even in mixed and irrelevant preference settings, highlighting the effectiveness of our proposed framework in diverse conditions.

| Model | Method | Robustness Error ($\downarrow$) |
| :--- | :--- | :--- |
| **Llama3-8B** | Naive Prompting | 20.9 |
| | Pref-Aligner (ours) | **18.1** |
| **Llama3-70B** | Naive Prompting | 9.0 |
| | Pref-Aligner (ours) | **6.5** |
| **Mixtral-8x7B** | Naive Prompting | 26.1 |
| | Pref-Aligner (ours) | **18.9** |
| **Gemma-2-9B** | Naive Prompting | 12.6 |
| | Pref-Aligner (ours) | **6.8** |

*Table: Robustness Error comparison between Naive Prompting (Zero-Shot) and Pref-Aligner across four models. Pref-Aligner consistently reduces robustness error across all models, achieving a minimum relative reduction of 13\% (Llama3-70B) and up to 46\% (Gemma-2-9B).*

<br>

| Method | Relevant ($\downarrow$) | Mixed ($\downarrow$) | Irrelevant ($\downarrow$) |
| :--- | :--- | :--- | :--- |
| Naive Prompting | 5.6 | 6.9 | 5.5 |
| Pref-Aligner (ours) | **1.1** | **1.2** | **1.2** |

*Table: Breakage Rate: Pref-Aligner Results compared to Zero-Shot for Llama-70B in three preference relevance settings. Pref-Aligner shows significant performance improvement over naive across all settings. Also, this performance remains consistent irrespective of preference setting.*

## Conclusion


