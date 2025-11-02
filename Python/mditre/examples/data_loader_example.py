"""
Example: Using MDITRE Data Loader System

This example demonstrates the modular data loading system for various
microbiome data formats.
"""

import numpy as np
import sys
sys.path.insert(0, 'd:/Github/mditre')

from mditre.data_loader import (
    DataLoaderRegistry,
    PickleDataLoader,
    DADA2Loader,
    TransformPipeline,
    NormalizeTransform,
    FilterLowAbundance,
    CLRTransform,
    create_data_loader,
    create_stratified_loaders,
    compute_phylo_distance_matrix,
    get_otu_embeddings
)


def example_1_list_available_loaders():
    """Example 1: List all registered data loaders"""
    print("=" * 80)
    print("Example 1: Available Data Loaders")
    print("=" * 80)
    
    available_loaders = DataLoaderRegistry.list_loaders()
    print(f"Registered loaders ({len(available_loaders)}):")
    for loader_name in available_loaders:
        loader_class = DataLoaderRegistry.get_loader(loader_name)
        print(f"  - {loader_name}: {loader_class.__name__}")
    print()


def example_2_load_pickle_data():
    """Example 2: Load data from pickle format"""
    print("=" * 80)
    print("Example 2: Load Pickle Data")
    print("=" * 80)
    
    # Method 1: Direct instantiation
    loader = PickleDataLoader('path/to/data.pkl')
    print(f"Created loader: {loader.__class__.__name__}")
    
    # Method 2: Using registry
    loader2 = DataLoaderRegistry.create_loader(
        'pickle',
        'path/to/data.pkl',
        config={'min_samples_per_subject': 3}
    )
    print(f"Created via registry: {loader2.__class__.__name__}")
    print()


def example_3_data_transformations():
    """Example 3: Apply data transformations"""
    print("=" * 80)
    print("Example 3: Data Transformations")
    print("=" * 80)
    
    # Create synthetic data
    n_subjects, n_otus, n_timepoints = 10, 50, 15
    X = np.random.rand(n_subjects, n_otus, n_timepoints) * 100
    
    print(f"Original data shape: {X.shape}")
    print(f"Original data range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Single transformation
    normalize = NormalizeTransform()
    X_norm = normalize(X)
    print(f"\nAfter normalization:")
    print(f"  Range: [{X_norm.min():.4f}, {X_norm.max():.4f}]")
    print(f"  Sum along OTU axis: {X_norm.sum(axis=1)[0, 0]:.4f}")
    
    # Pipeline of transformations
    pipeline = TransformPipeline([
        NormalizeTransform(),
        FilterLowAbundance(min_abundance=0.001, min_prevalence=0.1),
        CLRTransform()
    ])
    
    print(f"\nApplying pipeline: {pipeline}")
    X_transformed = pipeline(X)
    print(f"  Transformed shape: {X_transformed.shape}")
    print(f"  Transformed range: [{X_transformed.min():.2f}, {X_transformed.max():.2f}]")
    print()


def example_4_pytorch_datasets():
    """Example 4: Create PyTorch datasets and loaders"""
    print("=" * 80)
    print("Example 4: PyTorch Datasets")
    print("=" * 80)
    
    # Synthetic data
    n_subjects, n_otus, n_timepoints = 100, 50, 15
    X = np.random.rand(n_subjects, n_otus, n_timepoints).astype(np.float32)
    y = np.random.randint(0, 2, n_subjects)
    mask = np.ones((n_subjects, n_timepoints), dtype=np.float32)
    
    print(f"Data: {n_subjects} subjects, {n_otus} OTUs, {n_timepoints} timepoints")
    print(f"Labels: {(y==1).sum()} positive, {(y==0).sum()} negative")
    
    # Create single data loader
    train_loader = create_data_loader(
        X, y, mask,
        batch_size=16,
        shuffle=True
    )
    print(f"\nCreated DataLoader with {len(train_loader)} batches")
    
    # Test iteration
    for batch_idx, batch in enumerate(train_loader):
        data = batch['data']
        labels = batch['label']
        masks = batch['mask']
        print(f"  Batch {batch_idx}: data={data.shape}, labels={labels.shape}, mask={masks.shape}")
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    # Create stratified train/val split
    print("\nCreating stratified train/val split...")
    train_loader, val_loader = create_stratified_loaders(
        X, y, mask,
        train_ratio=0.8,
        batch_size=16
    )
    # Static analyzer doesn't recognize Dataset.__len__, but it exists at runtime
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")  # type: ignore
    print(f"  Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")  # type: ignore
    print()


def example_5_phylogenetic_processing():
    """Example 5: Process phylogenetic information"""
    print("=" * 80)
    print("Example 5: Phylogenetic Processing")
    print("=" * 80)
    
    from ete3 import Tree
    
    # Create simple tree
    tree_str = "((A:1,B:1):0.5,(C:0.8,D:0.8):0.7);"
    tree = Tree(tree_str)
    
    print("Phylogenetic tree:")
    print(tree)
    
    # Compute distance matrix
    dist_matrix = compute_phylo_distance_matrix(tree)
    print(f"\nDistance matrix shape: {dist_matrix.shape}")
    print("Distance matrix:")
    print(dist_matrix)
    
    # Generate embeddings
    embeddings = get_otu_embeddings(tree, method='distance', emb_dim=2)
    print(f"\nOTU embeddings shape: {embeddings.shape}")
    print("Embeddings:")
    print(embeddings)
    print()


def example_6_complete_workflow():
    """Example 6: Complete data loading workflow"""
    print("=" * 80)
    print("Example 6: Complete Workflow")
    print("=" * 80)
    
    # Simulate loading and preprocessing
    print("Step 1: Load raw data")
    n_subjects, n_otus, n_timepoints = 50, 100, 20
    X_raw = np.random.rand(n_subjects, n_otus, n_timepoints) * 100
    y = np.random.randint(0, 2, n_subjects)
    print(f"  Loaded {n_subjects} subjects with {n_otus} OTUs")
    
    print("\nStep 2: Apply preprocessing pipeline")
    pipeline = TransformPipeline([
        NormalizeTransform(),
        FilterLowAbundance(min_abundance=0.01, min_prevalence=0.2),
    ])
    X_processed = pipeline(X_raw)
    print(f"  After filtering: {X_processed.shape[1]} OTUs retained")
    
    print("\nStep 3: Create train/val/test splits")
    # Split into train+val and test
    n_test = int(0.2 * n_subjects)
    test_indices = np.random.choice(n_subjects, n_test, replace=False)
    train_val_indices = np.array([i for i in range(n_subjects) if i not in test_indices])
    
    X_train_val = X_processed[train_val_indices]
    y_train_val = y[train_val_indices]
    X_test = X_processed[test_indices]
    y_test = y[test_indices]
    
    # Split train_val into train and val
    train_loader, val_loader = create_stratified_loaders(
        X_train_val, y_train_val,
        train_ratio=0.8,
        batch_size=8
    )
    test_loader = create_data_loader(X_test, y_test, batch_size=8, shuffle=False)
    
    # Static analyzer doesn't recognize Dataset.__len__, but it exists at runtime
    print(f"  Train: {len(train_loader.dataset)} samples")  # type: ignore
    print(f"  Val: {len(val_loader.dataset)} samples")  # type: ignore
    print(f"  Test: {len(test_loader.dataset)} samples")  # type: ignore
    
    print("\nStep 4: Ready for model training!")
    print("  Use train_loader for training")
    print("  Use val_loader for validation")
    print("  Use test_loader for final evaluation")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MDITRE Data Loader System Examples")
    print("=" * 80 + "\n")
    
    example_1_list_available_loaders()
    example_2_load_pickle_data()
    example_3_data_transformations()
    example_4_pytorch_datasets()
    example_5_phylogenetic_processing()
    example_6_complete_workflow()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
