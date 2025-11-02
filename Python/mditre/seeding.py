"""
MDITRE Seeding Utilities

Deterministic seed generation using the seedhash library.
This ensures reproducibility across different runs and experiments.

The seeding process works as follows:
1. A seed_string (e.g., MDITRE master seed + experiment name) is hashed using MD5
2. The MD5 hash is converted to a seed_number (master_seed) - a deterministic integer
3. This master_seed is used to generate additional repeatable random seeds
4. Seeds can be used to initialize random number generators (Python, NumPy, PyTorch)

Example workflow:
    >>> seed_gen = MDITRESeedGenerator()
    >>> master_seed = seed_gen.generate_seeds(1)[0]  # Get the master seed (from hash)
    >>> set_random_seeds(master_seed)  # Set all RNGs
    >>> # Or generate multiple seeds for different components:
    >>> seeds = seed_gen.generate_seeds(5)  # [seed1, seed2, seed3, seed4, seed5]
"""

from typing import List, Optional
try:
    from seedhash import SeedHashGenerator
except ImportError:
    raise ImportError(
        "seedhash is required for MDITRE seeding. "
        "Install it with: pip install git+https://github.com/melhzy/seedhash.git#subdirectory=Python"
    )

# Master seed string for MDITRE project
# This string is hashed (MD5) to produce a deterministic seed_number
MDITRE_MASTER_SEED = "MDITRE: Scalable and Interpretable Machine Learning for Predicting Host Status from Temporal Microbiome Dynamics"


class MDITRESeedGenerator:
    """
    MDITRE seed generator using deterministic hashing.
    
    This class wraps the SeedHashGenerator to provide MDITRE-specific
    seeding functionality with a standard master seed string.
    
    The seed generation process:
    1. seed_string (master seed + optional experiment name) → MD5 hash
    2. MD5 hash → seed_number (master_seed) - a deterministic large integer
    3. seed_number → sequence of random seeds via internal PRNG
    
    All seeds are deterministic and reproducible given the same seed_string.
    
    Attributes:
        master_seed (str): The master seed string for MDITRE
        seed_string (str): The full seed string (master + experiment name)
        generator (SeedHashGenerator): The underlying seed generator
        
    Example:
        >>> from mditre.seeding import MDITRESeedGenerator, set_random_seeds
        >>> 
        >>> # Create generator with default MDITRE master seed
        >>> seed_gen = MDITRESeedGenerator()
        >>> 
        >>> # Get the master_seed (first seed from hash)
        >>> master_seed = seed_gen.generate_seeds(1)[0]
        >>> print(f"Master seed: {master_seed}")  # e.g., 951483900
        >>> 
        >>> # Use master_seed to set all RNGs
        >>> set_random_seeds(master_seed)
        >>> 
        >>> # Or generate multiple seeds for different components
        >>> seeds = seed_gen.generate_seeds(5)
        >>> # seeds[0] for data splitting, seeds[1] for model init, etc.
        >>> 
        >>> # Create generator with custom experiment name
        >>> seed_gen = MDITRESeedGenerator(experiment_name="experiment_v1")
        >>> exp_seeds = seed_gen.generate_seeds(3)
        >>> 
        >>> # Get the hash for reproducibility tracking
        >>> print(seed_gen.get_hash())
        'a1b2c3d4e5f6...'
    """
    
    def __init__(
        self,
        experiment_name: Optional[str] = None,
        min_value: int = 0,
        max_value: int = 2147483647
    ):
        """
        Initialize the MDITRE seed generator.
        
        Args:
            experiment_name: Optional experiment identifier to append to master seed.
                           If None, uses only the master seed.
            min_value: Minimum value for random seed range (default: 0)
            max_value: Maximum value for random seed range (default: 2^31-1)
            
        Raises:
            TypeError: If experiment_name is not a string or None
            ValueError: If min_value >= max_value
        """
        self.master_seed = MDITRE_MASTER_SEED
        
        # Construct the full seed string
        if experiment_name is not None:
            if not isinstance(experiment_name, str):
                raise TypeError("experiment_name must be a string or None")
            if not experiment_name.strip():
                raise ValueError("experiment_name cannot be empty")
            self.seed_string = f"{self.master_seed}::{experiment_name}"
        else:
            self.seed_string = self.master_seed
        
        # Initialize the generator
        self.generator = SeedHashGenerator(
            self.seed_string,
            min_value=min_value,
            max_value=max_value
        )
    
    def generate_seeds(self, count: int) -> List[int]:
        """
        Generate a list of deterministic random seeds.
        
        Args:
            count: Number of seeds to generate
            
        Returns:
            List of integer seeds
            
        Raises:
            TypeError: If count is not an integer
            ValueError: If count is not positive
            
        Example:
            >>> seed_gen = MDITRESeedGenerator()
            >>> seeds = seed_gen.generate_seeds(10)
            >>> len(seeds)
            10
        """
        return self.generator.generate_seeds(count)
    
    def get_hash(self) -> str:
        """
        Get the MD5 hash of the seed string.
        
        This hash can be used for reproducibility tracking and verification.
        
        Returns:
            MD5 hash as hexadecimal string
            
        Example:
            >>> seed_gen = MDITRESeedGenerator()
            >>> hash_value = seed_gen.get_hash()
            >>> len(hash_value)
            32
        """
        return self.generator.get_hash()
    
    def get_seed_info(self) -> dict:
        """
        Get comprehensive information about the seed generator.
        
        Returns:
            Dictionary containing:
                - seed_string: The full seed string used (master + experiment name)
                - hash: MD5 hash of the seed string (32 character hex string)
                - seed_number: Large integer derived from hash (master_seed)
                - min_value: Minimum seed value for generated seeds
                - max_value: Maximum seed value for generated seeds
                - master_seed: The MDITRE master seed string (constant)
                
        Note:
            The seed_number is the deterministic integer derived from the MD5 hash
            of the seed_string. This is what we call the "master_seed" when we 
            generate the first seed: master_seed = generate_seeds(1)[0]
                
        Example:
            >>> seed_gen = MDITRESeedGenerator(experiment_name="test")
            >>> info = seed_gen.get_seed_info()
            >>> print(f"Hash: {info['hash']}")
            >>> print(f"Seed number: {info['seed_number']}")  # Large int from MD5
        """
        return {
            "seed_string": self.seed_string,
            "hash": self.get_hash(),
            "seed_number": self.generator.seed_number,
            "min_value": self.generator.min_value,
            "max_value": self.generator.max_value,
            "master_seed": self.master_seed,
        }


def get_mditre_seeds(
    count: int,
    experiment_name: Optional[str] = None,
    min_value: int = 0,
    max_value: int = 2147483647
) -> List[int]:
    """
    Convenience function to generate MDITRE seeds.
    
    This is a wrapper around MDITRESeedGenerator for quick seed generation.
    
    Args:
        count: Number of seeds to generate
        experiment_name: Optional experiment identifier
        min_value: Minimum value for random seed range
        max_value: Maximum value for random seed range
        
    Returns:
        List of integer seeds
        
    Example:
        >>> from mditre.seeding import get_mditre_seeds
        >>> 
        >>> # Generate 5 seeds for a specific experiment
        >>> seeds = get_mditre_seeds(5, experiment_name="baseline_model")
        >>> print(seeds)
        [123456789, 987654321, ...]
    """
    generator = MDITRESeedGenerator(
        experiment_name=experiment_name,
        min_value=min_value,
        max_value=max_value
    )
    return generator.generate_seeds(count)


def set_random_seeds(seed: int):
    """
    Set random seeds for all common libraries to ensure reproducibility.
    
    This function sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA if available)
    
    For PyTorch, also configures:
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    
    These settings ensure fully deterministic behavior across runs.
    
    Args:
        seed: Integer seed value (typically the master_seed from generate_seeds(1)[0])
        
    Example:
        >>> from mditre.seeding import MDITRESeedGenerator, set_random_seeds
        >>> 
        >>> # Generate master_seed from MDITRE seed string
        >>> seed_gen = MDITRESeedGenerator()
        >>> master_seed = seed_gen.generate_seeds(1)[0]  # e.g., 951483900
        >>> 
        >>> # Set all random number generators
        >>> set_random_seeds(master_seed)
        >>> 
        >>> # Now all random operations are reproducible
        >>> import numpy as np
        >>> import torch
        >>> print(np.random.randint(0, 100))  # Same value every time
        >>> print(torch.randint(0, 100, (1,)))  # Same value every time
    """
    import random
    import numpy as np
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed


# Module-level documentation is complete in the module docstring above
