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
