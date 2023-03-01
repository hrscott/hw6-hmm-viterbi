import numpy as np
class HiddenMarkovModel:
    """_summary_
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_probabilities: np.ndarray, transition_probabilities: np.ndarray, emission_probabilities: np.ndarray):
        """_summary_

        Args:
            observation_states (np.ndarray): _description_
            hidden_states (np.ndarray): _description_
            prior_probabilities (np.ndarray): _description_
            transition_probabilities (np.ndarray): _description_
            emission_probabilities (np.ndarray): _description_
        """             
        self.observation_states = observation_states
        self.observation_states_dict = {observation_state: observation_state_index \
                                  for observation_state_index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {hidden_state_index: hidden_state \
                                   for hidden_state_index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_probabilities= prior_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities