import copy
import numpy as np

class ViterbiAlgorithm:
    """An implementation of the Viterbi algorithm for Hidden Markov Models."""

    def __init__(self, hmm_object):
        """Initializes the ViterbiAlgorithm object.

        Args:
            hmm_object (HiddenMarkovModel): An object representing the Hidden Markov Model.
        """
        self.hmm_object = hmm_object

    def best_hidden_state_sequence(self, decode_observation_states: np.ndarray) -> np.ndarray:
        """Finds the most likely sequence of hidden states that generated a given sequence of observation states.

        Args:
            decode_observation_states (np.ndarray): A NumPy array representing the observation states to decode.

        Returns:
            np.ndarray: A NumPy array representing the most likely sequence of hidden states.
        """

        ## INITIALIZATION
        # Initialize two arrays to store the hidden state sequence that returns the maximum probability
        path = np.zeros((len(decode_observation_states), len(self.hmm_object.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hmm_object.hidden_states))]
        best_path = np.zeros((len(decode_observation_states), len(self.hmm_object.hidden_states)))

        # Compute the initial delta value and scale factor
        first_observation_state_index = self.hmm_object.observation_states_dict[decode_observation_states[0]]
        delta = self.hmm_object.prior_probabilities * self.hmm_object.emission_probabilities[:, first_observation_state_index]
        if np.sum(delta) == 0:
            delta = np.full_like(delta, np.finfo(float).eps)

        #scaling
        #scale_factor = 1.0 / np.sum(delta)
        #delta /= scale_factor

        ## RECURSION
        # For each observation state to decode, select the hidden state sequence with the highest probability (i.e., Viterbi trellis)
        for trellis_node in range(1, len(decode_observation_states)):
            # Compute the product of delta and transition and emission probabilities
            product_of_delta_and_transition_emission = np.multiply(delta, self.hmm_object.transition_probabilities.T)
            product_of_delta_and_transition_emission = np.multiply(product_of_delta_and_transition_emission, self.hmm_object.emission_probabilities[:, self.hmm_object.observation_states_dict[decode_observation_states[trellis_node]]])

            # Update delta and scale
            delta = np.max(product_of_delta_and_transition_emission, axis=1)
            if np.sum(delta) == 0:
                delta = np.full_like(delta, np.finfo(float).eps)

            #scaling
            #scale_factor = 1.0 / np.sum(delta)
            #delta /= scale_factor

            # Select the hidden state sequence with the maximum probability
            best_path[trellis_node,:] = np.argmax(product_of_delta_and_transition_emission, axis=1)

            # Update the best path and current path for the next trellis node
            for hidden_state in range(len(self.hmm_object.hidden_states)):
                best_hidden_state_index = int(best_path[trellis_node, hidden_state])
                path[trellis_node, hidden_state] = best_hidden_state_index
                best_path[trellis_node, hidden_state] = delta[best_hidden_state_index] * product_of_delta_and_transition_emission[best_hidden_state_index, hidden_state]

        ## TERMINATION
        # Select the last hidden state given the best path (i.e., maximum probability)
        best_hidden_state_path = np.zeros(len(decode_observation_states), dtype='U11')
        best_hidden_state_path[-1] = self.hmm_object.hidden_states[int(np.argmax(delta))]

        # Backtrack to get the sequence of hidden states that generated the observations
        for trellis_node in range(len(decode_observation_states)-2, -1, -1):
            best_hidden_state_index = path[trellis_node+1, int(best_path[trellis_node+1, best_hidden_state_index])]
            #best_hidden_state_index = int(best_path[trellis_node, self.hmm_object.hidden_states.index(best_hidden_state_path[trellis_node+1])])
            best_hidden_state_index = int(best_path[trellis_node, np.argmax(delta)])
            best_hidden_state_path[trellis_node] = self.hmm_object.hidden_states[best_hidden_state_index]
        return best_hidden_state_path