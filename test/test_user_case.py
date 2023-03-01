"""
UCSF BMI203: Biocomputing Algorithms
Author: Henry Scott
Date: 02/2023
Program: Biophysics
Description: My best shot
"""
import pytest
import numpy as np
from models.hmm import HiddenMarkovModel
from models.decoders import ViterbiAlgorithm
from numpy.testing import assert_array_almost_equal


def test_use_case_lecture():
    """_summary_
    """
    # index annotation observation_states=[i,j]    
    observation_states = ['committed','ambivalent'] # A graduate student's dedication to their rotation lab
    
    # index annotation hidden_states=[i,j]
    hidden_states = ['R01','R21'] # The NIH funding source of the graduate student's rotation project 

    # PONDERING QUESTION: How would a user define/compute their own HMM instantiation inputs to decode the hidden states for their use case observations?
    use_case_one_data = np.load('./data/UserCase-Lecture.npz')

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
    print(use_case_decoded_hidden_states)
    print(use_case_one_data['hidden_states'])
    #assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])


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
    #assert np.alltrue(use_case_decoded_hidden_states == use_case_one_data['hidden_states'])

#Test Case: Scalability of the Algorithm
#Hypothesis: The Viterbi algorithm can handle large sequences of observation states efficiently.
def test_viterbi_algorithm_handles_large_sequences():
    # Initialize a Hidden Markov Model with pre-defined probabilities and observations
    observation_states = np.array(['A', 'B', 'C', 'D', 'E'])
    hidden_states = np.array(['X', 'Y'])
    prior_probabilities = np.array([0.5, 0.5])
    transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    emission_probabilities = np.array([[0.1, 0.4, 0.2, 0.2, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]])
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_probabilities, transition_probabilities, emission_probabilities)

    # Initialize a ViterbiAlgorithm object
    viterbi_algorithm = ViterbiAlgorithm(hmm)

    # Define a large sequence of observation states
    decode_observation_states = np.random.choice(observation_states, size=1000)

    # Compute the most likely sequence of hidden states
    predicted_hidden_states = viterbi_algorithm.best_hidden_state_sequence(decode_observation_states)

    # Check if the predicted hidden state sequence has the same length as the sequence of observation states
    assert len(predicted_hidden_states) == len(decode_observation_states)


#Hypothesis: The Viterbi algorithm can accurately predict hidden state sequences even when the Hidden Markov Model is trained on a small amount of data
def test_viterbi_algorithm_with_single_observation_state():
    """Tests the Viterbi algorithm's ability to predict a single hidden state when given a single observation state.
    """
    # Set up test data
    observation_states = np.array(['X'])
    hidden_states = np.array(['A'])
    prior_probabilities = np.array([1.0])
    transition_probabilities = np.array([[1.0]])
    emission_probabilities = np.array([[1.0]])

    # Create HiddenMarkovModel object
    hmm = HiddenMarkovModel(observation_states, hidden_states, prior_probabilities, transition_probabilities, emission_probabilities)

    # Create ViterbiAlgorithm object
    viterbi = ViterbiAlgorithm(hmm)

    # Predict hidden state sequence for single observation state
    decoded_hidden_states = viterbi.best_hidden_state_sequence(observation_states)

    # Check that predicted hidden state sequence is correct
    assert np.array_equal(decoded_hidden_states, np.array(['A']))




    
 #tests in development...
    
# def test_viterbi_algorithm_handles_zero_emission_probabilities():
#     # Define an example Hidden Markov Model
#     observation_states = np.array(['A', 'B'])
#     hidden_states = np.array(['X', 'Y'])
#     prior_probabilities = np.array([0.5, 0.5])
#     transition_probabilities = np.array([[0.5, 0.5], [0.5, 0.5]])
#     emission_probabilities = np.array([[1, 0], [0, 0]])

#     # Initialize a ViterbiAlgorithm object with the HMM
#     hmm = HiddenMarkovModel(observation_states, hidden_states, prior_probabilities, transition_probabilities, emission_probabilities)
#     viterbi = ViterbiAlgorithm(hmm)

#     # Define an observation sequence with some emission probabilities set to zero
#     observation_sequence = np.array(['A', 'B', 'A', 'A'])

#     # Run the Viterbi algorithm to decode the observation sequence
#     predicted_hidden_state_sequence = viterbi.best_hidden_state_sequence(observation_sequence)

#     # Define the expected output sequence of hidden states
#     expected_hidden_state_sequence = np.array(['X', 'Y', 'Y', 'Y'])

#     # Check that the predicted and expected hidden state sequences are equal
#     np.testing.assert_array_equal(predicted_hidden_state_sequence, expected_hidden_state_sequence)



# def test_viterbi_algorithm_handles_nonuniform_prior_probabilities():
#     # Define an example Hidden Markov Model with nonuniform prior probabilities
#     observation_states = np.array(['A', 'B', 'C'])
#     hidden_states = np.array(['X', 'Y'])
#     prior_probabilities = np.array([0.8, 0.2])
#     transition_probabilities = np.array([[0.7, 0.3], [0.3, 0.7]])
#     emission_probabilities = np.array([[0.2, 0.8, 0], [0.5, 0, 0.5]])

#     # Initialize a ViterbiAlgorithm object with the HMM
#     hmm = HiddenMarkovModel(observation_states, hidden_states, prior_probabilities, transition_probabilities, emission_probabilities)
#     viterbi = ViterbiAlgorithm(hmm)

#     # Define an observation sequence
#     observation_sequence = np.array(['B', 'C', 'A'])

#     # Run the Viterbi algorithm to decode the observation sequence
#     predicted_hidden_state_sequence = viterbi.best_hidden_state_sequence(observation_sequence)

#     # Define the expected output sequence of hidden states
#     expected_hidden_state_sequence = np.array(['Y', 'Y', 'X'])

#     # Check that the predicted and expected hidden state sequences are equal
#     np.testing.assert_array_equal(predicted_hidden_state_sequence, expected_hidden_state_sequence)





# def test_regulatory_state_changes():
#     # Define the observation and hidden states for progenitor and primitive CMs
#     observation_states = ['regulatory', 'regulatory-potential']
#     hidden_states = ['encode_atac', 'atac']
    
#     # Load the data for progenitor and primitive CMs
#     progenitor_data = np.load("./data/ProjectDeliverable-ProgenitorCMs.npz")
#     primitive_data = np.load('./data/ProjectDeliverable-PrimitiveCMs.npz')
    
#     # Instantiate the HMM with progenitor CMs priors
#     progenitor_hmm = HiddenMarkovModel(observation_states, hidden_states,
#                                        progenitor_data['prior_probabilities'],
#                                        progenitor_data['transition_probabilities'],
#                                        progenitor_data['emission_probabilities'])
    
#     # Instantiate the Viterbi algorithm with the progenitor HMM
#     progenitor_viterbi = ViterbiAlgorithm(progenitor_hmm)
    
#     # Use the progenitor HMM to decode the hidden state sequence for the progenitor observation states
#     progenitor_decoded_states = progenitor_viterbi.best_hidden_state_sequence(progenitor_data['observation_states'])
    
#     # Instantiate the HMM with primitive CMs priors
#     primitive_hmm = HiddenMarkovModel(observation_states, hidden_states,
#                                       primitive_data['prior_probabilities'],
#                                       primitive_data['transition_probabilities'],
#                                       primitive_data['emission_probabilities'])
    
#     # Instantiate the Viterbi algorithm with the primitive HMM
#     primitive_viterbi = ViterbiAlgorithm(primitive_hmm)
    
#     # Use the primitive HMM to decode the hidden state sequence for the primitive observation states
#     primitive_decoded_states = primitive_viterbi.best_hidden_state_sequence(primitive_data['observation_states'])
    
#     # Check if regulatory state changes can be inferred from the decoded hidden state sequences
#     assert ('encode_atac' in progenitor_decoded_states[:5] and 'atac' in progenitor_decoded_states[5:])
#     assert ('atac' in primitive_decoded_states[:5] and 'encode_atac' in primitive_decoded_states[5:])




# def test_viterbi_algorithm_accuracy_with_missing_noisy_observations():
#     # Create a Hidden Markov Model with 3 equally probable hidden states and 2 equally probable observation states
#     observation_states = np.array(['0', '1',])
#     hidden_states = np.array(['A', 'B', 'C'])
#     prior_probabilities = np.full(len(hidden_states), 1/len(hidden_states))
#     transition_probabilities = np.full((len(hidden_states), len(hidden_states)), 1/len(hidden_states))
#     emission_probabilities = np.full((len(hidden_states), len(observation_states)), 1/len(observation_states))
#     hmm = HiddenMarkovModel(observation_states, hidden_states, prior_probabilities, transition_probabilities, emission_probabilities)

#     # Generate a random sequence of observation states with missing or noisy observations
#     true_hidden_states = []
#     observation_sequence = []
#     for i in range(100):
#         # Randomly choose a hidden state
#         hidden_state = np.random.choice(hmm.hidden_states, p=hmm.prior_probabilities)

#         # Randomly choose an observation state
#         observation_state = np.random.choice(hmm.observation_states, p=hmm.emission_probabilities[hmm.hidden_states_dict[hidden_state], :])

#         # Randomly skip or add a noisy observation
#         skip_observation = np.random.choice([True, False], p=[0.1, 0.9])
#         if not skip_observation:
#             observation_sequence.append(observation_state)
#             true_hidden_states.append(hidden_state)

#     # Use the Viterbi algorithm to decode the observation sequence
#     viterbi_algorithm = ViterbiAlgorithm(hmm)
#     predicted_hidden_states = viterbi_algorithm.best_hidden_state_sequence(np.array(observation_sequence))

#     # Calculate the accuracy of the predicted hidden states
#     correct_predictions = 0
#     for i in range(len(predicted_hidden_states)):
#         if predicted_hidden_states[i] == true_hidden_states[i]:
#             correct_predictions += 1
#     accuracy = correct_predictions / len(predicted_hidden_states)

#     # Check that the accuracy is above a certain threshold
#     assert accuracy >= 0.9


