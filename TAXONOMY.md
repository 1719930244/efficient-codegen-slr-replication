# Classification Taxonomy

This document describes the 24-category classification taxonomy used to organize 124 primary studies on efficiency in LLM-based code generation.

## Organizing Principle

The taxonomy follows the **LLM lifecycle**, from data preparation through training and inference to evaluation. Each primary study is assigned to one or more categories based on its primary contribution.

```
Lifecycle Stage          Research Question    Categories
─────────────────────────────────────────────────────────
Data Preparation         RQ1                  1a, 1b, 1d
Model Training           RQ2                  2a-2f
Inference Optimization   RQ3                  3a-3o
Evaluation               RQ4                  4a, 4b
```

---

## RQ1: Data Preparation

How do data-level strategies improve the efficiency of training code LLMs?

### 1a - Data Selection (15 studies)
Techniques for selecting high-quality subsets from large code corpora to reduce training cost while maintaining model quality. Includes influence-function-based pruning, embedding-based selection, and task-aware filtering.

### 1b - Data Quality Assessment & Synthesis (17 studies)
Methods for evaluating training data quality and generating synthetic data when natural corpora are insufficient. Includes corruption-based quality scoring, LLM-assisted data refinement, and domain-specific synthesis for low-resource languages (e.g., Verilog, RTL).

### 1d - Data Mixing (1 study)
Strategies for combining data from multiple sources, languages, or domains to optimize training data composition.

---

## RQ2: Model Training

How do training-level techniques reduce the cost of building code LLMs?

### 2a - Efficient Pre-training (4 studies)
Architectural and scheduling optimizations that reduce pre-training cost. Includes code-aware tokenizers, dynamic packing, and Mixture-of-Experts routing during training.

### 2b - Parameter-Efficient Fine-Tuning (10 studies)
Methods that update only a small fraction of model parameters during fine-tuning. Includes LoRA, QLoRA, prefix tuning, adapters, and domain-specific PEFT adaptations for code tasks.

### 2c - Knowledge Distillation (7 studies)
Transferring capabilities from large teacher models to smaller student models. Includes standard distillation, agent-aware distillation (compressing multi-agent pipelines), and reasoning distillation (transferring chain-of-thought capabilities).

### 2d - Curriculum Learning (2 studies)
Organizing training data from simple to complex examples to improve convergence speed. Includes difficulty-based ordering and context-progressive training for fill-in-the-middle tasks.

### 2f - RL-based Training (3 studies)
Reinforcement learning techniques applied during training. Distinguishes between RL for generation process efficiency (reducing token/compute cost during inference) and RL for generated code efficiency (improving runtime performance of output code).

---

## RQ3: Inference Optimization

How do inference-level techniques reduce the cost of code generation?

The taxonomy for RQ3 is organized around three multiplicative cost factors:

```
Total Inference Cost = Cost_per_token x N_tokens x K_calls
```

### Reducing Cost per Token

#### 3a - Speculative Decoding (3 studies)
Using a small draft model to generate candidate tokens verified by the target model, achieving lossless speedup through parallel verification.

#### 3b - Early Exit (5 studies)
Terminating computation at intermediate layers when confidence is sufficient, reducing per-token compute for simple generations.

#### 3c - Non-Autoregressive Generation (3 studies)
Generating multiple tokens in parallel rather than one at a time, trading some accuracy for significant latency reduction.

#### 3g - Post-Training Quantization (8 studies)
Reducing numerical precision of model weights (e.g., from 16-bit to 4-bit) to decrease memory consumption and potentially improve throughput.

#### 3h - Model Pruning (2 studies)
Removing redundant parameters or structures from trained models. Includes unstructured pruning and low-rank decomposition.

#### 3f - KV Cache Optimization (3 studies)
Optimizing the key-value cache used during autoregressive generation. Includes incremental cache reuse for code editing, anchor attention for structural tokens, and predictive cache population.

### Reducing Token Count

#### 3d - Prompt Compression (7 studies)
Reducing input token counts while preserving information needed for accurate generation. Includes structure-aware compression, AST-level importance weighting, and symbolic token replacement.

#### 3e - Context Pruning (9 studies)
Selecting relevant context from repositories for code completion. Includes hierarchical pruning, retrieval-aware skipping, and dependency-based context filtering.

#### 3l - Code-Specific Optimization (8 studies)
Leveraging properties unique to code (syntax, structure, formatting conventions) for optimization. Includes grammar redesign for LLM consumption and formatting-aware token reduction.

#### 3n - Prompt Engineering (6 studies)
Optimizing prompt design for efficiency. Includes evolutionary prompt optimization, difficulty-based prompt strategy selection, and energy-aware prompt evaluation.

#### 3o - Chain-of-Thought Optimization (5 studies)
Reducing the token cost of reasoning chains while maintaining their benefits. Includes adaptive detail adjustment, compressed reasoning drafts, and reasoning distillation to smaller models.

#### 3j - Adaptive Sampling (9 studies)
Optimizing test-time compute allocation across generated candidates. Includes confidence-based early stopping, compute-optimal allocation based on difficulty, and multi-objective hyperparameter tuning.

### Reducing Invocation Count

#### 3i - Model Routing (11 studies)
Directing queries to appropriately-sized models based on task complexity. Includes complexity-based routing, cascading with self-testing, hint-based shepherding, and control models that suppress unnecessary invocations.

#### 3k - Multi-Agent Orchestration (8 studies)
Optimizing multi-agent code generation pipelines. Includes topology optimization via RL, confidence-based agent gating, and fast/slow reasoning separation.

### System-Level

#### 3m - System-Level Serving (5 studies)
Deployment infrastructure optimization. Includes SLA-aware batch sizing, context-aware model eviction, and runtime engine comparison for code LMs.

---

## RQ4: Evaluation

How is efficiency evaluated in LLM-based code generation research?

### 4a - Benchmark Design (2 studies)
Design and development of benchmarks that include efficiency dimensions.

### 4b - Empirical Study (19 studies)
Systematic empirical evaluations that compare techniques, establish baselines, or analyze evaluation practices across the field. Includes quantization evaluations, PEFT comparisons, energy measurements, and prompt interaction studies.
