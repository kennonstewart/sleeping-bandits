# AGENT INSTRUCTIONS: Unlearning-Bandit Simulation Suite Setup

Your primary goal is to initialize the repository for the **Unlearning-Bandit Simulation Suite**. You will create the necessary directory structure and populate it with Python files containing the class and method skeletons for the three core modules described in the research paper.

The following steps are provided as guidelines, but feel free to adapt them as the situation requires.

---

### **Step 1: Create Directory Structure**

First, create the following directories and empty `__init__.py` files to establish the Python package structure:

```bash
mkdir -p unlearning_bandits/unlearning_bandits
mkdir -p unlearning_bandits/experiments
mkdir -p unlearning_bandits/config
mkdir -p unlearning_bandits/tests

touch unlearning_bandits/unlearning_bandits/__init__.py
touch unlearning_bandits/tests/__init__.py
```

### Step 2: Create the Experiment Configuration File

Create a YAML file that will hold the parameters for an experiment run. This file allows a researcher to easily configure a simulation.

File: unlearning_bandits/config/simulation_config.yaml
```
# Configuration for an Unlearning Bandit Simulation

# --- Stream Parameters ---
# Total number of rounds (T) in the event stream.
total_rounds: 10000

# Probability of a DELETE event at any given round.
# The probability of an INSERT is (1 - deletion_probability).
deletion_probability: 0.1

# --- Bandit & Model Parameters ---
# The set of actions available to the agent.
# Corresponds to FAST INSERT, FAST DELETE, FULL RETRAIN[cite: 50].
actions:
  - 'FAST_INSERT'
  - 'FAST_DELETE'
  - 'FULL_RETRAIN'

# Deletion capacity (m): the number of deletions allowed before a full retrain is enforced[cite: 51].
deletion_capacity: 100

# Fixed cost (X) incurred by the FULL_RETRAIN action[cite: 51].
retrain_cost: 50.0

# --- Privacy Parameters ---
# Epsilon (ε) for differential privacy noise calibration[cite: 55, 62].
epsilon: 1.0
```
### Step 3: Skeleton for Module 1: Event Stream Generator

Create the Python file for the module that generates the stream of INSERT and DELETE commands.

File: unlearning_bandits/unlearning_bandits/stream_generator.py

```
import numpy as np
from typing import Iterator, Tuple

class EventStreamGenerator:
    """
    Generates a stream of events for the unlearning bandit simulation.
    """
    def __init__(self, total_rounds: int, deletion_probability: float):
        """
        Initializes the generator.

        Args:
            total_rounds (int): The total number of events to generate (T).
            deletion_probability (float): The probability of a DELETE event.
        """
        self.total_rounds = total_rounds
        self.deletion_probability = deletion_probability

    def generate_stream(self) -> Iterator[Tuple[str, dict]]:
        """
        Yields a stream of events and associated data.

        Each event is a tuple containing the context ('INSERT' or 'DELETE') 
        and a placeholder for the data associated with that event.

        Yields:
            Iterator[Tuple[str, dict]]: An iterator over (context, data) tuples.
        """
        for t in range(self.total_rounds):
            # In a real scenario, data would be features/labels. Here, it's a placeholder.
            mock_data = {'id': t, 'features': np.random.rand(10)}
            if np.random.rand() < self.deletion_probability:
                yield 'DELETE', mock_data
            else:
                yield 'INSERT', mock_data
```
### Step 4: Skeleton for Module 2: Pluggable Model Handler

Create the Python file for the model handler. This should be an abstract base class to ensure that any model used in the simulation implements the required "arms".

File: unlearning_bandits/unlearning_bandits/model_handler.py
```
from abc import ABC, abstractmethod

class BaseModelHandler(ABC):
    """
    Abstract Base Class for a pluggable machine learning model.

    This class defines the interface for the core operations that the bandit
    agent can choose as its "arms": fast updates, fast deletions,
    and full model retraining[cite: 50].
    """

    @abstractmethod
    def fast_insert(self, data: dict) -> float:
        """
        Performs a fast, incremental update of the model.
        Returns the prediction loss after the update.
        """
        pass

    @abstractmethod
    def fast_delete(self, data: dict) -> float:
        """
        Performs a fast, efficient unlearning operation.
        Returns the prediction loss after the deletion.
        """
        pass

    @abstractmethod
    def full_retrain(self) -> float:
        """
        Performs a full, costly retraining of the model from scratch.
        Returns the prediction loss after the retraining.
        """
        pass

    @abstractmethod
    def predict(self, data: dict) -> float:
        """
        Calculates the instantaneous prediction loss for a given data point[cite: 53].
        """
        pass

import numpy as np
```
### Step 5: Skeleton for Module 3: Unlearn-UCB Bandit Agent

Create the Python file for the 

Unlearn-UCB agent. This class will contain the logic for action selection and state updates as described in the paper.

File: unlearning_bandits/unlearning_bandits/bandit_agent.py
```
class UnlearnUCBAgent:
    """
    Implements the Unlearn-UCB algorithm from the paper[cite: 65].

    This agent treats model update strategies as arms and uses a UCB policy
    to minimize regret while respecting a deletion capacity[cite: 8, 9].
    """
    def __init__(self, actions: list, epsilon: float, deletion_capacity: int):
        """
        Initializes the Unlearn-UCB agent.

        Args:
            actions (list): List of possible action names.
            epsilon (float): The privacy parameter ε for noise calibration[cite: 55].
            deletion_capacity (int): The hard deletion limit 'm'[cite: 51].
        """
        self.actions = actions
        self.epsilon = epsilon
        self.deletion_capacity = deletion_capacity

        # Track statistics per context ('INS', 'DEL') and action ('a')[cite: 59].
        self.q_values = {ctx: {act: 0.0 for act in actions} for ctx in ['INSERT', 'DELETE']}
        self.arm_pulls = {ctx: {act: 0 for act in actions} for ctx in ['INSERT', 'DELETE']}
        self.total_pulls = 0
        self.deletion_counter = 0

    def select_action(self, context: str, t: int) -> str:
        """
        Selects an action based on the UCB formula and current context[cite: 67, 69].

        Enforces a hard switch to FULL_RETRAIN if deletion capacity is met[cite: 52].

        Args:
            context (str): The current context, either 'INSERT' or 'DELETE'.
            t (int): The current timestep, t.

        Returns:
            str: The name of the action to be played.
        """
        # Enforce hard switch to FULL RETRAIN if deletion capacity is exceeded[cite: 52].
        if self.deletion_counter >= self.deletion_capacity:
            return 'FULL_RETRAIN'

        # UCB calculation for each admissible arm.
        ucb_scores = {}
        for action in self.actions:
            if self.arm_pulls[context][action] == 0:
                return action  # Play each arm at least once.

            # Simplified UCB formula from Algorithm 1[cite: 68].
            # In a full implementation, privacy noise variance (σ_ε^2) would be here.
            mean_reward = self.q_values[context][action]
            exploration_bonus = np.sqrt(2 * np.log(t) / self.arm_pulls[context][action])
            ucb_scores[action] = mean_reward + exploration_bonus

        return max(ucb_scores, key=ucb_scores.get)


    def update_statistics(self, context: str, action: str, reward: float):
        """
        Updates the agent's internal statistics after an action is played[cite: 70].

        Args:
            context (str): The context in which the action was played.
            action (str): The action that was played.
            reward (float): The reward received from the environment.
        """
        # Update counts and mean reward (Q-value)[cite: 59].
        self.arm_pulls[context][action] += 1
        n = self.arm_pulls[context][action]
        old_q = self.q_values[context][action]
        self.q_values[context][action] = old_q + (reward - old_q) / n

        # Update deletion counter and reset if necessary[cite: 70].
        if action == 'FAST_DELETE':
            self.deletion_counter += 1
        elif action == 'FULL_RETRAIN':
            self.deletion_counter = 0
```
### Step 6: Skeleton for the Main Simulation Runner

Finally, create a script that will orchestrate the simulation. It will load the configuration, initialize the modules, and run the main event loop.

File: unlearning_bandits/experiments/run_simulation.py

```
# This file will eventually contain the main simulation loop.
# For now, it serves as a placeholder for the complete experiment script.

def main():
    """
    Main function to run the unlearning bandit simulation.
    """
    print("Setting up simulation...")
    # 1. Load configuration from `config/simulation_config.yaml`.

    # 2. Initialize EventStreamGenerator.

    # 3. Initialize a concrete implementation of BaseModelHandler.

    # 4. Initialize the UnlearnUCBAgent.

    # 5. Loop for t = 1 to T[cite: 66]:
    #    a. Get context (c_t) from the stream generator[cite: 67].
    #    b. Agent selects an action (A_t)[cite: 69].
    #    c. Model handler executes the action.
    #    d. Calculate reward (r_t) based on loss and action cost[cite: 53, 54].
    #    e. Agent updates statistics with the received reward[cite: 70].

    # 6. Log results and generate plots for regret analysis.
    print("Simulation setup complete. Ready for implementation.")

if __name__ == '__main__':
    main()
```
