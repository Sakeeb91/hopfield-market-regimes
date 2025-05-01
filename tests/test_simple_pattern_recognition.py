import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tempfile

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hopfield_network import HopfieldNetwork


class TestSimplePatternRecognition(unittest.TestCase):
    """
    Simple demonstration of Hopfield Network for pattern recognition.
    This test shows how the Hopfield Network can recognize distorted patterns.
    """
    
    def setUp(self):
        """Set up the test case with some simple patterns."""
        # Define a 5x5 letter 'X' pattern
        self.x_pattern = np.array([
            [1, -1, -1, -1, 1],
            [-1, 1, -1, 1, -1],
            [-1, -1, 1, -1, -1],
            [-1, 1, -1, 1, -1],
            [1, -1, -1, -1, 1]
        ]).flatten()
        
        # Define a 5x5 letter 'O' pattern
        self.o_pattern = np.array([
            [-1, 1, 1, 1, -1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [1, -1, -1, -1, 1],
            [-1, 1, 1, 1, -1]
        ]).flatten()
        
        # Define a 5x5 letter 'T' pattern
        self.t_pattern = np.array([
            [1, 1, 1, 1, 1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1],
            [-1, -1, 1, -1, -1]
        ]).flatten()
        
        # Create a Hopfield Network with 25 neurons (5x5 grid)
        self.network = HopfieldNetwork(25)
        
        # Store the patterns
        self.network.store_pattern(self.x_pattern, label="X")
        self.network.store_pattern(self.o_pattern, label="O")
        self.network.store_pattern(self.t_pattern, label="T")
    
    def test_pattern_recognition(self):
        """Test that the network can recognize patterns with noise."""
        # Create a noisy 'X' pattern (flip 20% of the bits)
        noisy_x = self.x_pattern.copy()
        indices = np.random.choice(25, size=5, replace=False)  # 5 out of 25 is 20%
        for idx in indices:
            noisy_x[idx] = -noisy_x[idx]
        
        # Run the network dynamics
        final_state, iterations, energy_values = self.network.run_dynamics(
            noisy_x, 
            max_iterations=100, 
            async_updates=True
        )
        
        # Find the closest stored pattern
        closest_idx, overlap, pattern = self.network.get_closest_pattern(final_state)
        
        # The closest pattern should be 'X'
        self.assertEqual(self.network.pattern_labels[closest_idx], "X")
        
        # The overlap should be high (at least 0.9)
        self.assertGreaterEqual(overlap, 0.9)
        
        # Create a visualization of this test
        self._visualize_pattern_recognition(noisy_x, final_state)
    
    def _visualize_pattern_recognition(self, noisy_pattern, final_state):
        """
        Helper method to visualize the pattern recognition process.
        This is not an actual test but a visualization of the process.
        """
        # Don't generate plots during automated testing
        if 'CI' in os.environ:
            return
            
        # Create a figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot original patterns
        pattern_shapes = {
            'X': self.x_pattern.reshape(5, 5),
            'O': self.o_pattern.reshape(5, 5),
            'T': self.t_pattern.reshape(5, 5)
        }
        
        # Original patterns
        combined = np.vstack([
            np.hstack([pattern_shapes['X'], np.ones((5, 1)), pattern_shapes['O'], np.ones((5, 1)), pattern_shapes['T']])
        ])
        
        # Convert from bipolar (-1,1) to binary (0,1) for visualization
        combined_binary = (combined + 1) / 2
        
        axes[0].imshow(combined_binary, cmap='binary')
        axes[0].set_title('Stored Patterns (X, O, T)')
        axes[0].axis('off')
        
        # Noisy pattern
        noisy_binary = (noisy_pattern.reshape(5, 5) + 1) / 2
        axes[1].imshow(noisy_binary, cmap='binary')
        axes[1].set_title('Noisy Pattern (Input)')
        axes[1].axis('off')
        
        # Final state
        final_binary = (final_state.reshape(5, 5) + 1) / 2
        axes[2].imshow(final_binary, cmap='binary')
        axes[2].set_title('Recovered Pattern (Output)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the figure to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            temp_path = temp.name
            plt.savefig(temp_path)
            print(f"\nPattern recognition visualization saved to: {temp_path}")
            
        plt.close(fig)
        
    def test_multiple_pattern_recognition(self):
        """Test that the network can recognize multiple patterns."""
        # Create a dictionary to store results
        results = {}
        
        for pattern_name, pattern in [('X', self.x_pattern), ('O', self.o_pattern), ('T', self.t_pattern)]:
            # Create a noisy version (flip 20% of the bits)
            noisy_pattern = pattern.copy()
            indices = np.random.choice(25, size=5, replace=False)
            for idx in indices:
                noisy_pattern[idx] = -noisy_pattern[idx]
                
            # Classify the pattern
            result = self.network.classify_pattern(noisy_pattern)
            
            # Store the result
            results[pattern_name] = result
            
            # Check that it was classified correctly
            self.assertEqual(result['label'], pattern_name)
            self.assertGreaterEqual(result['overlap'], 0.8)
            
        # Print summary (not part of the test)
        for pattern_name, result in results.items():
            print(f"\nPattern: {pattern_name}")
            print(f"Classified as: {result['label']}")
            print(f"Overlap: {result['overlap']:.2f}")
            print(f"Iterations: {result['iterations']}")
            print(f"Final energy: {result['final_energy']:.2f}")


if __name__ == '__main__':
    unittest.main() 