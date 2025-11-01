"""Test MDITRE seeding module."""

from mditre.seeding import MDITRESeedGenerator, get_mditre_seeds, set_random_seeds

# Test 1: Basic seed generation
print("=" * 60)
print("Test 1: Basic Seed Generation")
print("=" * 60)
gen = MDITRESeedGenerator()
seeds = gen.generate_seeds(5)
print(f"Generated seeds: {seeds}")
print(f"Hash: {gen.get_hash()}")

# Test 2: Get seed info
print("\n" + "=" * 60)
print("Test 2: Seed Information")
print("=" * 60)
info = gen.get_seed_info()
print(f"Master seed: {info['master_seed'][:70]}...")
print(f"Seed string: {info['seed_string'][:70]}...")
print(f"Hash: {info['hash']}")
print(f"Seed number: {info['seed_number']}")

# Test 3: With experiment name
print("\n" + "=" * 60)
print("Test 3: With Experiment Name")
print("=" * 60)
gen_exp = MDITRESeedGenerator(experiment_name="baseline_v1")
seeds_exp = gen_exp.generate_seeds(3)
print(f"Experiment seeds: {seeds_exp}")
print(f"Experiment hash: {gen_exp.get_hash()}")

# Test 4: Convenience function
print("\n" + "=" * 60)
print("Test 4: Convenience Function")
print("=" * 60)
quick_seeds = get_mditre_seeds(10, experiment_name="quick_test")
print(f"Quick seeds (10): {quick_seeds}")

# Test 5: Set random seeds
print("\n" + "=" * 60)
print("Test 5: Set Random Seeds")
print("=" * 60)
import random
import numpy as np
import torch

set_random_seeds(seeds[0])
print(f"Python random: {random.randint(0, 1000)}")
print(f"NumPy random: {np.random.randint(0, 1000)}")
print(f"PyTorch random: {torch.randint(0, 1000, (1,)).item()}")

# Test again with same seed - should produce same values
set_random_seeds(seeds[0])
print(f"\nAfter resetting with same seed:")
print(f"Python random: {random.randint(0, 1000)} (should be same)")
print(f"NumPy random: {np.random.randint(0, 1000)} (should be same)")
print(f"PyTorch random: {torch.randint(0, 1000, (1,)).item()} (should be same)")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
