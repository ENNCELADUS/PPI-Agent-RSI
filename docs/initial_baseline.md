# Initial Baseline: Fixed Scientific Agent and PPI Predictor

The initial baseline can be a **fixed scientific agent built around an expert-designed workflow and a trained PPI predictor**, evaluated on held-out experimental data. Its purpose is to measure performance and identify failure modes before developing recursive self-improvement (RSI).

- **Task:** Rank candidate protein pairs under specified biological contexts and interventions, identifying interactions and intervention-sensitive PPIs.
- **System:** A pinned LLM follows an expert-defined workflow: obtain PPI predictions → retrieve contextual evidence → compare candidates → produce rankings and predicted responses. Prompts, tools, and predictor parameters remain fixed.
- **Data:** Use public information and designated private development data as inputs. Hold out unpublished experimental outcomes, including matched reference–intervention measurements where available. Exclude those outcomes from predictor training and workflow design.
- **Evaluation:** First assess predictions and rankings without feedback. Then reveal a limited number of existing measurements and let the same fixed procedure reassess the remaining candidates. This evaluates how effectively the agent uses new evidence without requiring new experiments.
- **Metrics:** Evaluate interaction-ranking quality, intervention-response prediction, calibration, and discoveries per revealed measurement. Report results separately for familiar versus held-out receptors or contexts, and for complete versus missing modalities.
- **Failure analysis:** Examine errors attributable to the predictor, context mismatch, conflicting evidence, missing modalities, and ineffective use of feedback. Compare against the predictor alone or inspect individual pipeline stages to identify bottlenecks.

**Expected deliverable:** A measured baseline and a reproducible set of failure cases indicating where workflow-level or model-level improvement is needed. Failure modes should be established through evaluation rather than assumed in advance.
