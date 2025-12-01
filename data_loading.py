"""
Data loading utilities for Matrix CLIP training.
Supports various data formats and storage structures.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Optional
import json


class MatrixPairDataset(Dataset):
    """
    Dataset for loading paired matrices (116×166 and 116×116).
    
    Supports multiple folder structures:
    1. Separate folders for each matrix type
    2. Paired files with matching names
    3. Single file containing both matrices
    """
    
    def __init__(
        self,
        data_dir: str,
        matrix1_subdir: str = "matrix1",
        matrix2_subdir: str = "matrix2",
        file_extension: str = ".npy",
        expected_shape1: tuple = (116, 116),
        expected_shape2: tuple = (116, 116),
        transform1=None,
        transform2=None,
        normalize: bool = True,
        sc_edge_dropout: float = 0.0
    ):
        """
        Args:
            data_dir: Root directory containing the data
            matrix1_subdir: Subdirectory name for 116×166 matrices
            matrix2_subdir: Subdirectory name for 116×116 matrices
            file_extension: File extension (.npy, .npz, .pt, .txt, .csv)
            transform1: Optional transform for matrix1
            transform2: Optional transform for matrix2
            normalize: Whether to normalize matrices (recommended)
        """
        self.data_dir = Path(data_dir)
        self.matrix1_dir = self.data_dir / matrix1_subdir
        self.matrix2_dir = self.data_dir / matrix2_subdir
        self.file_extension = file_extension
        self.expected_shape1 = expected_shape1
        self.expected_shape2 = expected_shape2
        self.transform1 = transform1
        self.transform2 = transform2
        self.normalize = normalize
        self.sc_edge_dropout = float(sc_edge_dropout)

        # If no explicit transform2 provided and a positive dropout is requested,
        # create a simple edge-dropout transform that zeroes entries with probability p.
        if (self.transform2 is None) and (self.sc_edge_dropout > 0.0):
            p = float(self.sc_edge_dropout)
            def edge_dropout_transform(matrix: np.ndarray) -> np.ndarray:
                # matrix is a 2D numpy array; apply independent dropout per entry
                mask = (np.random.rand(*matrix.shape) > p).astype(matrix.dtype)
                return matrix * mask
            self.transform2 = edge_dropout_transform

        # Get list of paired files
        self.file_pairs = self._get_file_pairs()
        
        if len(self.file_pairs) == 0:
            raise ValueError(f"No paired files found in {data_dir}")
        
        print(f"Found {len(self.file_pairs)} matrix pairs")
    
    def _get_file_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching pairs of matrix files."""
        pairs = []
        
        # Get all files from matrix1 directory
        matrix1_files = sorted(self.matrix1_dir.glob(f"*{self.file_extension}"))
        
        for m1_file in matrix1_files:
            # Look for corresponding matrix2 file with same base name
            base_name = m1_file.stem
            m2_file = self.matrix2_dir / f"{base_name}{self.file_extension}"
            
            if m2_file.exists():
                pairs.append((m1_file, m2_file))
            else:
                print(f"Warning: No matching matrix2 file for {m1_file.name}")
        
        return pairs
    
    def _load_matrix(self, file_path: Path) -> np.ndarray:
        """Load matrix from file based on extension."""
        if self.file_extension == ".npy":
            return np.load(file_path)
        elif self.file_extension == ".npz":
            data = np.load(file_path)
            # Assume the matrix is stored with key 'data' or first key
            key = 'data' if 'data' in data else list(data.keys())[0]
            return data[key]
        elif self.file_extension == ".pt":
            return torch.load(file_path).numpy()
        elif self.file_extension == ".txt":
            return np.loadtxt(file_path)
        elif self.file_extension == ".csv":
            # CSV with comma delimiter; assumes no header
            return np.loadtxt(file_path, delimiter=',')
        else:
            raise ValueError(f"Unsupported file extension: {self.file_extension}")
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to zero mean and unit variance."""
        mean = matrix.mean()
        std = matrix.std()
        if std > 0:
            return (matrix - mean) / std
        return matrix - mean
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        m1_file, m2_file = self.file_pairs[idx]
        
        # Load matrices
        matrix1 = self._load_matrix(m1_file)
        matrix2 = self._load_matrix(m2_file)
        
        # Validate shapes
        assert matrix1.shape == self.expected_shape1, f"Matrix1 has wrong shape: {matrix1.shape}, expected {self.expected_shape1}"
        assert matrix2.shape == self.expected_shape2, f"Matrix2 has wrong shape: {matrix2.shape}, expected {self.expected_shape2}"
        
        # Normalize if requested
        if self.normalize:
            matrix1 = self._normalize_matrix(matrix1)
            matrix2 = self._normalize_matrix(matrix2)
        
        # Apply transforms
        if self.transform1:
            matrix1 = self.transform1(matrix1)
        if self.transform2:
            matrix2 = self.transform2(matrix2)
        
        # Convert to tensors
        matrix1 = torch.from_numpy(matrix1).float()
        matrix2 = torch.from_numpy(matrix2).float()
        
        return matrix1, matrix2


class SingleFileMatrixDataset(Dataset):
    """
    Dataset for loading matrices from a single file containing all pairs.
    Useful when all data fits in memory.
    """
    
    def __init__(
        self,
        file_path: str,
        normalize: bool = True
    ):
        """
        Args:
            file_path: Path to file containing both matrices
                      For .npz: should have 'matrix1' and 'matrix2' keys
                      For .pt: should be a dict with 'matrix1' and 'matrix2'
            normalize: Whether to normalize matrices
        """
        self.file_path = Path(file_path)
        self.normalize = normalize
        
        # Load all data
        if self.file_path.suffix == ".npz":
            data = np.load(self.file_path)
            self.matrix1_data = data['matrix1']
            self.matrix2_data = data['matrix2']
        elif self.file_path.suffix == ".pt":
            data = torch.load(self.file_path)
            self.matrix1_data = data['matrix1'].numpy()
            self.matrix2_data = data['matrix2'].numpy()
        else:
            raise ValueError(f"Unsupported file type: {self.file_path.suffix}")
        
        # Validate shapes
        assert self.matrix1_data.shape[1:] == (116, 166), f"Matrix1 has wrong shape: {self.matrix1_data.shape}"
        assert self.matrix2_data.shape[1:] == (116, 116), f"Matrix2 has wrong shape: {self.matrix2_data.shape}"
        assert len(self.matrix1_data) == len(self.matrix2_data), "Mismatched number of samples"
        
        print(f"Loaded {len(self.matrix1_data)} matrix pairs from {file_path}")
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix to zero mean and unit variance."""
        mean = matrix.mean()
        std = matrix.std()
        if std > 0:
            return (matrix - mean) / std
        return matrix - mean
    
    def __len__(self) -> int:
        return len(self.matrix1_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix1 = self.matrix1_data[idx].copy()
        matrix2 = self.matrix2_data[idx].copy()
        
        # Normalize if requested
        if self.normalize:
            matrix1 = self._normalize_matrix(matrix1)
            matrix2 = self._normalize_matrix(matrix2)
        
        # Convert to tensors
        matrix1 = torch.from_numpy(matrix1).float()
        matrix2 = torch.from_numpy(matrix2).float()
        
        return matrix1, matrix2


def create_dataloaders(
    train_dir: str,
    val_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Directory containing training data
        val_dir: Directory containing validation data (optional)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        **dataset_kwargs: Additional arguments for MatrixPairDataset
    
    Returns:
        train_loader, val_loader (val_loader is None if val_dir not provided)
    """
    # Create training dataset
    train_dataset = MatrixPairDataset(train_dir, **dataset_kwargs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for consistent batch size
    )
    
    # Create validation dataset if provided
    val_loader = None
    if val_dir is not None:
        val_dataset = MatrixPairDataset(val_dir, **dataset_kwargs)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


def save_matrix_pair(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    save_dir: str,
    filename: str,
    format: str = "npy"
):
    """
    Save a pair of matrices to disk.
    
    Args:
        matrix1: 116×166 matrix
        matrix2: 116×116 matrix
        save_dir: Directory to save files
        filename: Base filename (without extension)
        format: Save format ('npy', 'npz', 'pt')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "npy":
        # Save as separate files
        matrix1_dir = save_dir / "matrix1"
        matrix2_dir = save_dir / "matrix2"
        matrix1_dir.mkdir(exist_ok=True)
        matrix2_dir.mkdir(exist_ok=True)
        
        np.save(matrix1_dir / f"{filename}.npy", matrix1)
        np.save(matrix2_dir / f"{filename}.npy", matrix2)
    
    elif format == "npz":
        # Save in single compressed file
        np.savez_compressed(
            save_dir / f"{filename}.npz",
            matrix1=matrix1,
            matrix2=matrix2
        )
    
    elif format == "pt":
        # Save as PyTorch tensor
        torch.save({
            'matrix1': torch.from_numpy(matrix1),
            'matrix2': torch.from_numpy(matrix2)
        }, save_dir / f"{filename}.pt")


# Example usage and data structure documentation
RECOMMENDED_STRUCTURE = """
RECOMMENDED FOLDER STRUCTURE:
=============================

Option 1: Separate Folders (RECOMMENDED for many files)
--------------------------------------------------------
data/
├── train/
│   ├── matrix1/              # 116×166 matrices
│   │   ├── sample_000.npy
│   │   ├── sample_001.npy
│   │   ├── sample_002.npy
│   │   └── ...
│   └── matrix2/              # 116×116 matrices
│       ├── sample_000.npy
│       ├── sample_001.npy
│       ├── sample_002.npy
│       └── ...
└── val/
    ├── matrix1/
    │   ├── sample_000.npy
    │   └── ...
    └── matrix2/
        ├── sample_000.npy
        └── ...

Usage:
    train_loader, val_loader = create_dataloaders(
        train_dir="data/train",
        val_dir="data/val",
        batch_size=32
    )


Option 2: Single File (Good for smaller datasets that fit in RAM)
------------------------------------------------------------------
data/
├── train_data.npz            # Contains 'matrix1' and 'matrix2' arrays
└── val_data.npz

Usage:
    train_dataset = SingleFileMatrixDataset("data/train_data.npz")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


Option 3: Paired Files per Sample
-----------------------------------
data/
├── train/
│   ├── sample_000_m1.npy     # 116×166 matrix
│   ├── sample_000_m2.npy     # 116×116 matrix
│   ├── sample_001_m1.npy
│   ├── sample_001_m2.npy
│   └── ...
└── val/
    ├── sample_000_m1.npy
    ├── sample_000_m2.npy
    └── ...

(Would need custom loader - see MatrixPairDataset for adaptation)


FILE FORMATS SUPPORTED:
=======================
- .npy:  NumPy binary format (fast, efficient)
- .npz:  Compressed NumPy format (smaller files)
- .pt:   PyTorch tensor format
- .txt:  Plain text (human-readable but slower)

RECOMMENDED: Use .npy for individual files or .npz for single-file storage
"""

if __name__ == "__main__":
    print(RECOMMENDED_STRUCTURE)
