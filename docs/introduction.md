# Project Introduction and Research Direction

## Project Overview

This project focuses on cross-tissue membrane protein interactions (PPIs) and their functional regulation. It investigates context- and perturbation-aware multimodal prediction, evidence-constrained scientific decision-making, and recursive self-improvement driven by limited experimental feedback. The goal is to develop an integrated AI framework in which PPI prediction models and scientific agents work together to support biological discovery.

## Research Direction 1. Context- and Perturbation-Aware Multimodal Interaction and Local Response Modeling

Integrate protein sequences, structures, membrane topology, and tissue- and cell-level omics data to study representations that jointly capture local molecular compatibility and environmental conditions. Use these representations to predict candidate interactions, associated binding interfaces, and responses to perturbations such as drug treatments and mutations.

To address imbalances in data coverage and representation quality across modalities and tasks, develop a learning mechanism combining shared interaction representations with condition-specific prediction. Jointly use multigranular supervision from interaction pairs, local interfaces, and condition-dependent responses to improve predictive performance under sparse data and generalization across biological contexts.

## Research Direction 2. Evidence-Constrained Scientific Hypothesis Generation and Candidate Prioritization

Build a scientific agent for membrane protein functional discovery using pretrained large language models, PPI predictions, cross-tissue network analysis, and scientific knowledge retrieval.

Develop a structured representation of hypotheses in the form **interaction pair → biological context → perturbation condition → experimental readout**. Establish a reasoning mechanism jointly constrained by evidence provenance, conditions of applicability, and predictive uncertainty. The agent will assess interaction confidence, perturbation responses, and disease-relevant functional value to produce traceable candidate rankings and proposed validation strategies.

## Research Direction 3. Dual-Level Recursive Self-Improvement Driven by Limited Experimental Feedback

Develop a recursive self-improvement framework in which scientific agent workflows and PPI prediction models evolve together, addressing the sparsity, heterogeneity, and delays characteristic of experimental feedback.

Combine historical data with newly acquired experimental results to diagnose errors and automatically propose improvements to evidence organization, tool use, candidate selection, and validation strategies. Extend this optimization through controlled code modifications to multimodal fusion and model-training strategies.

Establish independent evaluation, version selection, and experience reuse mechanisms to continuously improve prediction reliability and scientific discovery efficiency with limited experimental feedback.
