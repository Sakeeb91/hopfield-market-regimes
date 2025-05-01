import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


class HopfieldNetwork:
    """
    Implementation of a Hopfield Network for pattern recognition and classification.
    
    Hopfield Networks are a form of recurrent artificial neural network that can
    serve as content-addressable memory systems, capable of recognizing and
    classifying patterns.
    """
    
    def __init__(self, num_neurons: int):
        """
        Initialize a Hopfield Network with a specified number of neurons.
        
        Args:
            num_neurons: The number of neurons in the network
        """
        self.num_neurons = num_neurons
        # Initialize weight matrix with zeros
        self.weights = np.zeros((num_neurons, num_neurons))
        # Track stored patterns for reference
        self.stored_patterns = []
        # Track pattern labels if provided
        self.pattern_labels = {}
    
    def store_pattern(self, pattern: np.ndarray, label: Optional[str] = None) -> None:
        """
        Store a pattern in the network using Hebbian learning rule.
        
        Args:
            pattern: Binary pattern to store (-1,1 or 0,1)
            label: Optional label for the pattern
        """
        if len(pattern) != self.num_neurons:
            raise ValueError(f"Pattern length ({len(pattern)}) must match number of neurons ({self.num_neurons})")
        
        # Convert to numpy array if not already
        pattern = np.array(pattern)
        
        # Ensure pattern is in bipolar form (-1, 1)
        if set(np.unique(pattern)).issubset({0, 1}):
            # Convert from binary (0,1) to bipolar (-1,1)
            pattern = 2 * pattern - 1
            
        # Outer product for weight update (excluding self-connections)
        pattern_matrix = np.outer(pattern, pattern)
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(pattern_matrix, 0)
        
        # Update weights
        self.weights += pattern_matrix / self.num_neurons
        
        # Store pattern for reference
        self.stored_patterns.append(pattern.copy())
        
        # Store label if provided
        if label is not None:
            self.pattern_labels[len(self.stored_patterns) - 1] = label
    
    def update_neuron(self, neuron_index: int, state: np.ndarray) -> int:
        """
        Update a single neuron based on current network state.
        
        Args:
            neuron_index: Index of the neuron to update
            state: Current state of the network
            
        Returns:
            New state of the updated neuron (-1 or 1)
        """
        # Calculate activation for the neuron
        activation = np.dot(self.weights[neuron_index], state)
        
        # Apply threshold function
        return 1 if activation >= 0 else -1
    
    def run_dynamics(self, initial_state: np.ndarray, 
                    max_iterations: int = 100, 
                    async_updates: bool = True,
                    convergence_threshold: int = 0) -> Tuple[np.ndarray, int, List[float]]:
        """
        Run the network dynamics until convergence or max iterations.
        
        Args:
            initial_state: Initial state of the network
            max_iterations: Maximum number of iterations to run
            async_updates: Whether to use asynchronous updates (True) or synchronous (False)
            convergence_threshold: Number of iterations without state change to consider converged
            
        Returns:
            Tuple of (final state, iterations used, energy values)
        """
        if len(initial_state) != self.num_neurons:
            raise ValueError(f"Initial state length ({len(initial_state)}) must match number of neurons ({self.num_neurons})")
        
        # Convert to numpy array if not already
        initial_state = np.array(initial_state)
        
        # Ensure state is in bipolar form (-1, 1)
        if set(np.unique(initial_state)).issubset({0, 1}):
            # Convert from binary (0,1) to bipolar (-1,1)
            initial_state = 2 * initial_state - 1
        
        state = initial_state.copy()
        prev_state = state.copy()
        
        # To track energy over iterations
        energy_values = [self.calculate_energy(state)]
        
        iterations = 0
        unchanged_iterations = 0
        
        while iterations < max_iterations:
            if async_updates:
                # Asynchronous updates (update one neuron at a time)
                for i in range(self.num_neurons):
                    # Randomly select a neuron to update
                    neuron_idx = np.random.randint(0, self.num_neurons)
                    state[neuron_idx] = self.update_neuron(neuron_idx, state)
            else:
                # Synchronous updates (update all neurons at once)
                new_state = np.zeros_like(state)
                for i in range(self.num_neurons):
                    new_state[i] = self.update_neuron(i, state)
                state = new_state
            
            # Calculate energy after update
            energy_values.append(self.calculate_energy(state))
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                unchanged_iterations += 1
                if unchanged_iterations >= convergence_threshold:
                    break
            else:
                unchanged_iterations = 0
                
            prev_state = state.copy()
            iterations += 1
            
        return state, iterations, energy_values
    
    def calculate_energy(self, state: np.ndarray) -> float:
        """
        Calculate the energy of the current network state.
        
        The energy function is E = -0.5 * state^T * W * state
        Lower energy means more stable state.
        
        Args:
            state: Current state of the network
            
        Returns:
            Energy value
        """
        return -0.5 * np.dot(np.dot(state, self.weights), state)
    
    def get_closest_pattern(self, state: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Find the stored pattern closest to the given state.
        
        Args:
            state: Current state of the network
            
        Returns:
            Tuple of (pattern index, overlap ratio, closest pattern)
        """
        if not self.stored_patterns:
            raise ValueError("No patterns have been stored in the network")
        
        # Ensure state is in bipolar form (-1, 1)
        if set(np.unique(state)).issubset({0, 1}):
            # Convert from binary (0,1) to bipolar (-1,1)
            state = 2 * state - 1
            
        max_overlap = -float('inf')
        closest_idx = -1
        
        for i, pattern in enumerate(self.stored_patterns):
            # Calculate dot product (measure of similarity)
            overlap = np.dot(state, pattern) / self.num_neurons
            if overlap > max_overlap:
                max_overlap = overlap
                closest_idx = i
        
        if closest_idx != -1:
            return closest_idx, max_overlap, self.stored_patterns[closest_idx]
        else:
            return None, 0, None
    
    def classify_pattern(self, probe: np.ndarray, **kwargs) -> dict:
        """
        Classify a probe pattern by running network dynamics and finding closest match.
        
        Args:
            probe: Pattern to classify
            **kwargs: Additional parameters for run_dynamics
            
        Returns:
            Dictionary with classification results
        """
        # Run network dynamics
        final_state, iterations, energy_values = self.run_dynamics(probe, **kwargs)
        
        # Find closest stored pattern
        closest_idx, overlap, pattern = self.get_closest_pattern(final_state)
        
        # Get the label if available
        label = self.pattern_labels.get(closest_idx, f"Pattern {closest_idx}")
        
        # Prepare result
        result = {
            "final_state": final_state,
            "closest_pattern_idx": closest_idx,
            "closest_pattern": pattern,
            "overlap": overlap,
            "label": label,
            "iterations": iterations,
            "energy_values": energy_values,
            "final_energy": energy_values[-1]
        }
        
        return result
    
    def visualize_weights(self, figsize=(10, 8), cmap='coolwarm'):
        """
        Visualize the weight matrix of the network.
        
        Args:
            figsize: Figure size
            cmap: Colormap for the heatmap
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.weights, cmap=cmap)
        plt.colorbar(label='Weight Value')
        plt.title('Hopfield Network Weight Matrix')
        plt.xlabel('Neuron j')
        plt.ylabel('Neuron i')
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_pattern(self, pattern, shape=None, figsize=(6, 6), cmap='binary'):
        """
        Visualize a pattern, optionally reshaping it.
        
        Args:
            pattern: The pattern to visualize
            shape: Optional shape to reshape the pattern (e.g., (10, 10) for 10x10 grid)
            figsize: Figure size
            cmap: Colormap for the image
        """
        plt.figure(figsize=figsize)
        
        # Convert from bipolar (-1,1) to binary (0,1) if needed
        display_pattern = pattern.copy()
        if -1 in display_pattern:
            display_pattern = (display_pattern + 1) / 2
        
        if shape is not None and np.prod(shape) == len(pattern):
            plt.imshow(display_pattern.reshape(shape), cmap=cmap)
        else:
            # If no shape provided or incorrect shape, show as 1D
            plt.plot(display_pattern, 'o-')
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            
        plt.title('Pattern Visualization')
        plt.tight_layout()
        return plt.gcf()
    
    def visualize_energy_landscape(self, energy_values, figsize=(10, 6)):
        """
        Visualize the energy landscape during network dynamics.
        
        Args:
            energy_values: List of energy values from run_dynamics
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        plt.plot(energy_values, 'o-', linewidth=2)
        plt.grid(True)
        plt.title('Energy Landscape During Network Dynamics')
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.tight_layout()
        return plt.gcf() 