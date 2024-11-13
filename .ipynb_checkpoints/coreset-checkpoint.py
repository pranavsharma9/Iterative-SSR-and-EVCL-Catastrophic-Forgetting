import torch
import pyro
import tyxe

import random
import copy
import functools
import heapq

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist

import pyro.distributions as dist
from tqdm import tqdm

def update_coreset(prev_coreset, train_loader, coreset_size, selection_method='random', curr_idx=0):
    if isinstance(train_loader, list) and selection_method == 'class_balanced':
        tasks_so_far_data = []
        assert curr_idx > 0
        for i in range(0, curr_idx):
            tasks_so_far_data.append(train_loader[i])
        
        # Create a class-balanced list of combined_data for the total size of coreset_size
        combined_data = []
        num_tasks = len(tasks_so_far_data)
        samples_per_task = coreset_size // num_tasks
        for curr_task_loader in tasks_so_far_data:
            curr_task_data = list(curr_task_loader.dataset)
            combined_data.extend(random.sample(curr_task_data, samples_per_task))
        
        remaining_samples = coreset_size - len(combined_data)
        if remaining_samples > 0:
            combined_data.extend(random.sample(list(tasks_so_far_data[-1].dataset), remaining_samples))
        return combined_data
        
    elif isinstance(train_loader, torch.utils.data.dataloader.DataLoader):
        curr_task_data = list(train_loader.dataset)
        curr_task_data = random.sample(curr_task_data, min(coreset_size*2, len(curr_task_data))) # truncating current tasks data to {coreset_size * 2} to make it lil faster
        combined_data = curr_task_data + prev_coreset if prev_coreset else curr_task_data
    
    if selection_method == 'random':
        curr_coreset = random.sample(combined_data, min(coreset_size, len(combined_data)))
    elif selection_method == 'k-center':
        curr_coreset = k_center_coreset(combined_data, coreset_size)
    elif selection_method == 'pca-k-center':
        curr_coreset = pca_k_center_coreset(combined_data, coreset_size)
    else:
        raise ValueError(f"Invalid selection method: {selection_method}")
    
    return curr_coreset

def k_center_coreset(data, coreset_size, via_pca=False):
    if not via_pca:
        data_array = np.array([x.cpu().numpy() for x, _ in data])
    else:
        data_array = data
        
    num_points = len(data_array)

    # Initialize the coreset with the first data point
    initial_index = 0  # deterministic start point
    coreset_indices = [initial_index]
    
    # Initialize the distances from the initial coreset point to all other points
    distances = np.full(num_points, np.inf)
    distances[initial_index] = 0
    for i in range(num_points):
        if i != initial_index:
            distances[i] = np.linalg.norm(data_array[i] - data_array[initial_index])
    
    # max-heap for maintaining max distances
    heap = [(-dist, i) for i, dist in enumerate(distances)]
    heapq.heapify(heap)

    # Iteratively select the farthest point from the current coreset
    while len(coreset_indices) < coreset_size:
        _, farthest_point_index = heapq.heappop(heap)
        if farthest_point_index not in coreset_indices:
            coreset_indices.append(farthest_point_index)
            
            # Update the distances and the heap for the remaining points
            for i in range(num_points):
                if i not in coreset_indices:
                    new_distance = np.linalg.norm(data_array[i] - data_array[farthest_point_index])
                    if new_distance < distances[i]:
                        distances[i] = new_distance
                        heap = [(-distances[j], j) for j in range(num_points) if j not in coreset_indices] # Rebuild the heap with updated distances
                        heapq.heapify(heap)

    if via_pca:
        return coreset_indices
    
    return [data[i] for i in coreset_indices]


def pca_k_center_coreset(data, coreset_size, n_components=20):
    data_array = np.array([x.cpu().numpy() for x, _ in data])
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data_array)
    
    coreset_indices = k_center_coreset(reduced_data, coreset_size, via_pca=True)
    return [data[i] for i in coreset_indices]