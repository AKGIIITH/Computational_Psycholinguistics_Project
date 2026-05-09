    # Conspiracy Belief Detection via Transformer Representations and Psycholinguistic Analysis

    ## Research Questions

    We plan to study whether conspiracy _belief_ can be detected from text, not
    whether a post discusses a conspiracy, but whether the author believes it.
    Believers and non-believers write about the same events using largely the same
    vocabulary; what differs is how they frame narratives and structure information. We investigate whether a fine-tuned language model can learn this signal, and whether the writing patterns of believers align with what Surprisal Theory and the Uniform Information Density hypothesis predict.

    ## Predictions

    - Contextual sequence-level representations will substantially outperform
    lexical models, since the belief signal is distributed across the full
    narrative rather than concentrated in specific words.

    - Pooling over the entire sequence will outperform privileging a single summary token, because belief is expressed through the accumulation of narrative roles across a post rather than any one phrase

    - Believer posts will show lower surprisal variance, consistent with UID —
    writing from a fixed internal conspiratorial schema (actor, action, effect,
    evidence, victim) produces more uniform information density than the varied
    rhetorical moves of a skeptic or analyst.

    - Narrative completeness, how fully a post populates actor, action, effect,          evidence, and victim roles will be a strong predictor of belief
    
    - Belief-relevant features will peak at mid-stack encoder layers rather than the
    final layer, consistent with probing studies of semantic and pragmatic
    properties in transformers.

    ## Methodology

    We will fine-tune DeBERTa-v3-large on the binary belief classification task, encoding each post with mean pooling over all token positions and passing the representation to a small classifier. We compare this against a range of baselines — from simple lexical models to classifiers built purely on psycholinguistic annotation counts — to understand what contextual representations add at each level. We also test variants of the neural system itself, varying the pooling strategy and how much of the encoder is updated, to identify which design choices actually drive performance. 

    To understand what the model has learned, we will compute per-post surprisal and test whether its distribution differs between classes as UID predicts, apply SHAP to identify which narrative properties drive model predictions, run layer-wise probing to locate where belief information is encoded, and use a masking experiment to test whether the model tracks the same structural dimensions the theory identifies.

    Multiple seeds will be ensembled; cross-validation used for model selection.

    ## Dataset

    PsyCoMark (SemEval-2026 Task 10, Subtask 2):

    - 4,316 training posts and 77 development posts labeled believer / non-believer / ambiguous
    - Token-level psycholinguistic annotations: Actor, Action, Effect, Evidence, Victim
    - ~44% word-type overlap between classes — the discriminative signal lies in
    narrative structure, not vocabulary
    - Source: Reddit posts spanning multiple conspiracy-related subreddits, covering
    a wide range of theories; annotations produced independently of belief labels
