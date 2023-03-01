[![hw6-hmm-viterbi](https://github.com/hrscott/hw6-hmm-viterbi/actions/workflows/python-package.yml/badge.svg)](https://github.com/hrscott/hw6-hmm-viterbi/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/hw6-hmm-viterbi/badge/?version=latest)](https://hw6-hmm-viterbi.readthedocs.io/en/latest/?badge=latest)

# Lab 6: Inferring CRE Selection Strategies from Chromatin Regulatory State Observations using a Hidden Markov Model and the Viterbi Algorithm

The aim of hw6 is to implement the Viterbi algorithm, a dynamic program that is a common decoder for Hidden Markov Models (HMMs). The lab is structured by training objective, project deliverables, and experimental deliverables:

**Training Objective**: Learn how to design reusable Python packages with automated code documentation and develop testable (user case) hypotheses using the Viterbi algorithm to decode the best path of hidden states for a sequence of observations. 

**Project Deliverable**: Produce a simple report for *functional characterization* inferred from a binary regulatory observation state pattern across cardiac developmental timepoints. 

**Experimental Deliverable**: Construct a positive control library for massively parallel reporter assays (MPRAs) and CRISPRi/a experiments in primitive and progenitor cardiomyocytes (i.e., cardiogenomics).

## Key Words
Chromatin; histones; nucleosomes; genomic element; accessible chromatin; chromatin states; genomic annotation; candidate cis-regulatory element (cCRE); Hidden Markov Model (HMM); ENCODE; ChromHMM; cardio-genomics; congenital heart disease(CHD); TBX5

# Project Overview

The hw6 project consists of **research training data** used to guide *functional characterization* experiments for assessing the regulatory state of non-protein coding regions in cardiac disease (i.e., congenital heart disease (CHD) or cardiometabolic disorders (CMD)) and development (i.e., embryonic). For the lab's experimental design, we selected non-protein coding regions that are speculated to be "active" on both day seven and fifteen in the cardiac (development) dataset/model for progenitor and primitive cardiomyocytes (CMs).  

**Vocabulary**: We define a *cis*-candidate regulatory element (i.e., **cCRE**) as a non-protein-coding region with supporting evidence that it regulates transcription in the orientation of a target gene. In hw6, a **cCRE** is specific to a cell-type and/or timepoint (i.e., gestation). By contrast, a candidate regulatory element (i.e., **CRE**) is a genomic element with a less specific cell or timepoint context, as it is identified from a low-resolution assay (i.e., ATAC-seq or ChIP-seq) and/or inferred (i.e., ChromHMM) to regulate expression.

For more background information, please refer to the [lab 6 write-up](https://docs.google.com/document/d/18bpiEfD2IOL2NSkw_Ol10hzQKK_vaxo3axNLldRSHmA/edit?usp=sharing). 

In this project 6, we will exploit the biological intuition that CREs can be inferred from binary regulatory **observation states**, which we categorize as "regulatory" and "regulatory potential". 

```python
observation_states = ['regulatory', 'regulatory_potential'] # observed regulatory activity in the TBX5 TAD of cardiomyocytes
```

Specifically, we can use (maximum) likelihood to infer the underlying sequence of hidden states from a pattern of binary state observations across the _**TBX5**_ topologically associated domain (**TAD**). Accordingly, the choice of **hidden states** defines a scientific hypothesis, namely in the way we defined CREs for the experiment. We then employ these same choices for both progenitor and primitive CMs.

**Hidden States** represent two different CRE selection strategies:

 1. **encode_atac**: ENCODE cCREs intersected with accessible chromatin regions identified using ATAC-seq.

 2. **atac**: Accessible chromatin regions selected using ATAC-seq without intersecting ENCODE cCREs.

```python
hidden_states = ['encode_atac', 'atac'] # The two cCRE selection strategies, in order
```

**Testable Hypothesis Question**: Can an HMM instantiated from progenitor CMs in the _**TBX5**_ TAD be used to decode the regulatory observation states for primitive CMs in the same genomic region (i.e., the _**TBX5**_ TAD)? 

Prior **observation states**, **transitions**, and **emission** probabilities were computed for the progenitor CMs. Specifically, we created a *sliding window* of size *60 kilobases* (no overlaps) over the *TBX5* TAD to define the inputs of an instantiated HMM class we make for you.

In similar wording, the HHM prior probabilities, observation and hidden states, and transition and emission probabilities are **pre-computed** for you. We define these inputs in the unit test hw6-hmm-viterbi/test/test_project_deliverable.py function called *test_deliverable*. 

If you open test_project_deliverable.py, you will see how the project deliverable HMM and Viterbi algorithm are already instantiated.   

```python
def test_deliverable():
    """_summary_
    """
    # index annotation observation_states=[i,j] 
    observation_states = ['regulatory', 'regulatory-potential'] # observed regulatory activity in the TBX5 TAD of cardiomyocytes
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['encode-atac', 'atac'] # In order of the two cCRE selection strategies (encode_atac, atac)

    # Import the HMM input data for progenitor cardiomyocytes (prefix: prog_cm)
    prog_cm_data = np.load('./data/ProjectDeliverable-ProgenitorCMs.npz')

    # Instantiate submodule class models.HiddenMarkovModel with progenitor cardiomyocytes
    # observation and hidden states, and prior, transition, and emission probabilities.
    prog_cm_hmm_object = HiddenMarkovModel(observation_states,
                                           hidden_states,
                                           prog_cm_data['prior_probabilities'], #  prior probabilities of hidden states in the order specified in the hidden_states list
                                           prog_cm_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                                           prog_cm_data['emission_probabilities'])  # emission_probabilities[hidden_states[i],:][:,observation_states[j]]

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prog_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states (i.e., CRE selection strategy) for the progenitor CMs and evaluate the model performace
    evaluate_viterbi_decoder_using_observation_states_of_prog_cm = prog_cm_viterbi_instance.best_hidden_state_sequence(prog_cm_data['observation_states'])
    
    # Evaluate the accuracy of using the progenitor cardiomyocyte HMM and Viterbi algorithm to decode the progenitor CM's CRE selection strategies
    # NOTE: Model is expected to perform with 80% accuracy
    assert np.sum(prog_cm_data['hidden_states'] == evaluate_viterbi_decoder_using_observation_states_of_prog_cm)/len(prog_cm_data['observation_states']) == 0.8

    ### Evaluate Primitive Cardiomyocyte Regulatory Observation Sequence ###
    # Import primitive cardiomyocyte data (prefix: prim_cm)
    prim_cm_data = np.load('./data/ProjectDeliverable-PrimitiveCMs.npz')

    # Instantiate submodule class models.ViterbiAlgorithm with the progenitor cardiomyocyte's HMM
    prim_cm_viterbi_instance = ViterbiAlgorithm(prog_cm_hmm_object)

    # Decode the hidden states of the primitive cardiomyocyte's regulatory observation states
    decoded_hidden_states_for_observed_states_of_prim_cm = prim_cm_viterbi_instance.best_hidden_state_sequence(prim_cm_data['observation_states'])
    assert np.sum(prim_cm_data['hidden_states'] == decoded_hidden_states_for_observed_states_of_prim_cm)/len(prim_cm_data['observation_states']) == 0.8
    
```

# Project and Training Deliverables

We recommend completing these deliverables in the listed order:

- [X] Implement the Viterbi algorithm in models.decoders.ViterbiAlgorithm found hw6-hmm-viterbi/src/models/decoders.py.

- [ ] Complete the docstrings in submodules models.hmm and models.decoders by following the pep257 format including pep484 type hints. 
    - Please note, Visual Studio Code supports an extension called [autoDocstring](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/pep257.md), which will help by creating an automated template for each class method and function.

- [X] Develop **two** user case unit tests that follow a testable hypothesis that you design for the Viterbi algorithm. State your hypothesis in the documentation.
    - For example, in the lecture, we reviewed an example where the observations were the dedication of a rotating graduate student, and the hidden states were the rotation project's NIH funding.

**Decision Questions, Reminders, & Important Notes**

- First focus on understanding and then implementing the Viterbi algorithm (piece meal style) with **simple** unit tests (i.e., assert statements) as checkpoints for edge and user cases.

- The use case reviewed in lecture can be found at the path hw6-hmm-viterbi/test/test_user_case.py in the pytest function **test_use_case_lecture**. Here is the link to the presentation titled ["Hidden Markov Models Inferring Academic Success"](https://docs.google.com/presentation/d/1-iE4dxesqii9MoBC5x3jcZ3tv7kLjSqpeB1Xfeeb5vQ/edit#slide=id.g205428ee7e4_0_515) (refer to slides 13-74).

The unit test inputs and (simple) assert checks for **user case one** have already been defined for you as an example.

```python
def test_user_case_one():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['on-time','late'] 

    # index annotation hidden_states=[i,j]
    hidden_states = ['no-traffic','traffic']

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-One.npz')

    # Instantiate submodule class models.HiddenMarkovModel with
    # observation and hidden states and prior, transition, and emission probabilities.
    use_case_one_hmm = HiddenMarkovModel(observation_states,
                                         hidden_states,
                      use_case_one_data['prior_probabilities'], # prior probabilities of hidden states in the order specified in the hidden_states list
                      use_case_one_data['transition_probabilities'], # transition_probabilities[:,hidden_states[i]]
                      use_case_one_data['emission_probabilities']) # emission_probabilities[hidden_states[i],:][:,observation_states[j]]
    
    # Instantiate submodule class models.ViterbiAlgorithm using the use case one HMM 
    use_case_one_viterbi = ViterbiAlgorithm(use_case_one_hmm)

     # Check if use case one HiddenMarkovAlgorithm instance is inherited in the subsequent ViterbiAlgorithm instance
    assert use_case_one_viterbi.hmm_object.observation_states == use_case_one_hmm.observation_states
    assert use_case_one_viterbi.hmm_object.hidden_states == use_case_one_hmm.hidden_states

    assert np.allclose(use_case_one_viterbi.hmm_object.prior_probabilities, use_case_one_hmm.prior_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.transition_probabilities, use_case_one_hmm.transition_probabilities)
    assert np.allclose(use_case_one_viterbi.hmm_object.emission_probabilities, use_case_one_hmm.emission_probabilities)

    # TODO: Check HMM dimensions and ViterbiAlgorithm
    
    # Find the best hidden state path for our observation states
    use_case_decoded_hidden_states = use_case_one_viterbi.best_hidden_state_sequence(use_case_one_data['observation_states'])
    assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])

```

**Design Checkpoint Question**: What additional edge cases will a user with a specific use case (i.e., testable hypothesis) might want to consider for models.decoders.ViterbiAlgorithm?

## ReadTheDocs

- [ ] Build the project deliverable's ReadTheDoc (RTD).

- [X] **Short Answer**: Evaluate the project deliverable and briefly answer the speculative questions in the RTD, with an eye to the project's limitations as related to the theory, model design, experimental data (i.e., biology and technology). Please navigate to hw6-hmm-viterbi/docs/index.rst and answer the questions under the sections titled **Functional Characterization Report**. 

```Markdown
1. Speculate how the progenitor cardiomyocyte Hidden Markov Model and primitive cardiomyocyte regulatory observations and inferred hidden states might change if the model design's sliding window (default set to 60 kilobases) were to increase or decrease?

2. How would you recommend integrating additional genomics data (i.e., histone and transcription factor ChIP-seq data) to update or revise the progenitor cardiomyocyte Hidden Markov Model? In your updated/revised model, how would you define the observation and hidden states, and the prior, transition, and emission probabilities? Using the updated/revised design, what new testable hypotheses would you be able to evaluate and/or disprove?

3. Following functional characterization (i.e., MPRA or CRISPRi/a) of progenitor and primitive cardiomyocytes, consider all possible scenarios for recommending how to update or revise our genomic annotation for *cis*-candidate regulatory elements (cCREs) and candidate regulatory elements (CREs)?

```

- We recommend answers between 2-6 sentences. It is OK if you are not familiar already with this biological user case; you can receive full points for your best-effort answer.


To jumpstart your ReadTheDocs, we recommend reviewing the [minimal-viable-package](https://github.com/bmi203-2023/minimal-viable-package). Please note, you will need to create a ReadTheDocs account and connect your forked repository.

# Software Development (provided for you)

Please note, a conda environment YML file is available in hw6-hmm-viterbi/env to provide all library and environment dependencies for you.

```bash
$ cd hw6-hmm-viterbi/env
$ conda env create -f hw6-hmm-viterbi-viterbi.yml
$ conda activate hw6-hmm-viterbi-viterbi.yml
```

In addtion, a working pyproject.toml, Git Actions main.yml workflow, and ReadTheDocs YML file are provided for your hw6 continuous development and integration. 

* For more information and guidance on sofware development, we recommend reviewing the [minimal-viable-package](https://github.com/bmi203-2023/minimal-viable-package).

# Grading Point Awards

Please note a complete submission will include a fully hosted ReadTheDocs with automated code documentation and your speculative answers to the inferred limitations of HW6.

- Implement submodule class models.decoders.ViterbiAlgorithm. [4 points]

- Complete a unit test for *test_user_case_one* in `hw6-hmm-viterbi/test/test_user_case.py`. [1 point] 

- Design **two** novel/independent unit tests for your own use case (i.e., testable hypothesis) in `hw6-hmm-viterbi/test/test_user_case.py`. [2 points]

- Complete docstring documentation for submodule classes and methods `models.decoders.ViterbiAlgorithm` and `models.hmm.HiddenMarkovModel`, and unit test functions for user cases and project deliverables. [1 point]

- Complete a unit test for the **project deliverable** in `hw6-hmm-viterbi/test/test_project_deliverable.py`. [1 point]

- Host your hw6-hmm-viterbi-viterbi documentation in ReadTheDocs and answer (speculative) questions (2-6 sentences) for the **project deliverable**. [1 point]
