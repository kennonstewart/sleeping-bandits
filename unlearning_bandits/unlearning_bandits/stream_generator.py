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
