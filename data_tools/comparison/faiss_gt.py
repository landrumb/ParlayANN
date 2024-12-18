#!/usr/bin/env python3

import faiss
import numpy as np
import argparse

def load_fbin(file_path, dimension):
    """Load vectors from a .fbin file."""
    length, dimension = np.fromfile(file_path, dtype='int32', count=2)
    print(f"Length: {length}, Dimension: {dimension}")
    vectors = np.fromfile(file_path, dtype='float32', offset=8)
    if vectors.size % dimension != 0:
        raise ValueError(f"File size is not a multiple of dimension {dimension}")
    return vectors.reshape(-1, dimension)

def main():
    parser = argparse.ArgumentParser(description="FAISS Exact Nearest Neighbor Search")
    parser.add_argument('--dataset', required=True, help='Path to dataset .fbin file')
    parser.add_argument('--queries', required=True, help='Path to queries .fbin file')
    parser.add_argument('--dimension', type=int, required=True, help='Dimensionality of vectors')
    parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to retrieve')
    parser.add_argument('--output', required=True, help='Path to output file for results')
    
    args = parser.parse_args()
    
    # Load dataset and queries
    print("Loading dataset...")
    dataset = load_fbin(args.dataset, args.dimension)
    print(f"Dataset size: {dataset.shape[0]} vectors")

    print("Loading queries...")
    queries = load_fbin(args.queries, args.dimension)
    print(f"Queries size: {queries.shape[0]} vectors")
    
    # Build the index
    print("Building the index...")
    index = faiss.IndexFlatL2(args.dimension)  # Exact search
    index.add(dataset)
    print(f"Total vectors in index: {index.ntotal}")
    
    # Perform the search
    print("Performing the search...")
    distances, indices = index.search(queries, args.k)
    
    # Save the results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'wb') as f:
        indices.tofile(f)
        distances.tofile(f)
    
    print("Search completed successfully.")

if __name__ == "__main__":
    import time
    
    start = time.time()
    main()
    print(f"Elapsed time: {time.time() - start:.2f} seconds")
