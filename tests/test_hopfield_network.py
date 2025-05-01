import unittest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hopfield_network import HopfieldNetwork


class TestHopfieldNetwork(unittest.TestCase):
    """Tests for the HopfieldNetwork class."""
    
    def test_initialization(self):
        """Test initialization of the HopfieldNetwork."""
        network = HopfieldNetwork(10)
        self.assertEqual(network.num_neurons, 10)
        self.assertEqual(network.weights.shape, (10, 10))
        self.assertEqual(len(network.stored_patterns), 0)
        
    def test_store_pattern(self):
        """Test storing a pattern in the network."""
        network = HopfieldNetwork(4)
        pattern = np.array([1, -1, 1, -1])
        network.store_pattern(pattern, label="test")
        
        # Check that the pattern was stored
        self.assertEqual(len(network.stored_patterns), 1)
        np.testing.assert_array_equal(network.stored_patterns[0], pattern)
        
        # Check that the label was stored
        self.assertEqual(network.pattern_labels[0], "test")
        
    def test_store_binary_pattern(self):
        """Test storing a binary pattern (0,1) in the network."""
        network = HopfieldNetwork(4)
        pattern = np.array([1, 0, 1, 0])  # Binary pattern
        network.store_pattern(pattern)
        
        # Check that the pattern was converted to bipolar (-1,1)
        self.assertEqual(len(network.stored_patterns), 1)
        np.testing.assert_array_equal(network.stored_patterns[0], np.array([1, -1, 1, -1]))
        
    def test_update_neuron(self):
        """Test updating a single neuron."""
        network = HopfieldNetwork(3)
        # Set up a simple weight matrix
        network.weights = np.array([
            [0, 1, -1],
            [1, 0, 1],
            [-1, 1, 0]
        ])
        
        state = np.array([1, -1, 1])
        
        # Test updating neuron 0
        new_state = network.update_neuron(0, state)
        self.assertEqual(new_state, -1)  # 1*(-1) + (-1)*1 = -2, which is negative, so output is -1
        
    def test_run_dynamics_simple(self):
        """Test running network dynamics with a simple example."""
        network = HopfieldNetwork(3)
        pattern = np.array([1, 1, -1])
        network.store_pattern(pattern)
        
        # Create a slightly perturbed pattern
        noisy_pattern = np.array([1, -1, -1])  # One bit flipped
        
        # Run dynamics
        final_state, _, _ = network.run_dynamics(noisy_pattern, async_updates=False)
        
        # Should converge to stored pattern
        np.testing.assert_array_equal(final_state, pattern)
        
    def test_calculate_energy(self):
        """Test energy calculation."""
        network = HopfieldNetwork(3)
        pattern = np.array([1, 1, -1])
        network.store_pattern(pattern)
        
        # Energy of stored pattern should be lower than random pattern
        energy_stored = network.calculate_energy(pattern)
        
        random_pattern = np.array([-1, -1, -1])
        energy_random = network.calculate_energy(random_pattern)
        
        self.assertLess(energy_stored, energy_random)
        
    def test_get_closest_pattern(self):
        """Test finding the closest stored pattern."""
        network = HopfieldNetwork(4)
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([-1, -1, 1, 1])
        
        network.store_pattern(pattern1, label="Pattern 1")
        network.store_pattern(pattern2, label="Pattern 2")
        
        # Test with a pattern closer to pattern1
        test_pattern = np.array([1, 1, -1, 1])  # One bit different from pattern1
        
        closest_idx, overlap, closest_pattern = network.get_closest_pattern(test_pattern)
        
        self.assertEqual(closest_idx, 0)  # Should match pattern1
        self.assertEqual(overlap, 0.75)   # 3/4 bits match
        np.testing.assert_array_equal(closest_pattern, pattern1)
        
    def test_classify_pattern(self):
        """Test pattern classification."""
        network = HopfieldNetwork(4)
        pattern1 = np.array([1, 1, -1, -1])
        pattern2 = np.array([-1, -1, 1, 1])
        
        network.store_pattern(pattern1, label="Pattern 1")
        network.store_pattern(pattern2, label="Pattern 2")
        
        # Test with a pattern closer to pattern2
        test_pattern = np.array([-1, -1, 1, -1])  # One bit different from pattern2
        
        result = network.classify_pattern(test_pattern)
        
        self.assertEqual(result["label"], "Pattern 2")
        self.assertEqual(result["closest_pattern_idx"], 1)
        self.assertEqual(result["overlap"], 0.75)
        
    def test_memory_capacity(self):
        """Test the storage capacity of the network."""
        # For a network of size N, it can store approximately 0.14N patterns
        N = 50
        network = HopfieldNetwork(N)
        
        # Store some random patterns
        num_patterns = int(0.14 * N)  # Theoretical limit
        
        for i in range(num_patterns):
            # Generate random bipolar pattern
            pattern = np.random.choice([-1, 1], size=N)
            network.store_pattern(pattern, label=f"Pattern {i}")
            
        # Now test recall for each pattern
        successful_recalls = 0
        
        for i in range(num_patterns):
            pattern = network.stored_patterns[i]
            
            # Add some noise (flip 10% of bits)
            noisy_pattern = pattern.copy()
            indices = np.random.choice(N, size=int(0.1 * N), replace=False)
            for idx in indices:
                noisy_pattern[idx] = -noisy_pattern[idx]
                
            # Run dynamics
            final_state, _, _ = network.run_dynamics(noisy_pattern)
            
            # Check if recall was successful (at least 90% of bits match)
            if np.sum(final_state == pattern) / N >= 0.9:
                successful_recalls += 1
                
        # Check that at least 75% of patterns were successfully recalled
        success_rate = successful_recalls / num_patterns
        self.assertGreaterEqual(success_rate, 0.75)
        
        
if __name__ == '__main__':
    unittest.main() 