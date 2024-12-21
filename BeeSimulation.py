import numpy as np
import time

def BeeSimulation(num_bees, flower_positions, max_visits, threshold_mean, threshold_stddev, decay_factor=None):
    """
    Simulate bee behavior and trapline formation based on flower arrangement and bee learning.
    
    Args:
        num_bees (int): Number of bees.
        flower_positions (array): 2D positions of flowers and nest (index 0 is the nest).
        max_visits (int): Maximum number of visits per bee.
        threshold_mean (float): Mean threshold for quality perception.
        threshold_stddev (float): Stddev for quality perception.
        decay_factor (float): Decay factor for adjusting probabilities, if any.
        
    Returns:
        tuple: Contains time spent, discoveries, visits, final states, and bee paths.
    """
    nestNum = len(flower_positions)  # Total sites including the nest
    bees = []

    # Distance matrix (pairwise Euclidean distances)
    distances = np.linalg.norm(flower_positions[:, None] - flower_positions, axis=2)

    # Initialize dynamic probabilities based on distance
    initial_probs = np.exp(-distances / np.max(distances))
    initial_probs /= initial_probs.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1

    for bee_idx in range(num_bees):
        bee = {
            'path': [],
            'times': [],
            'threshold': np.random.normal(threshold_mean, threshold_stddev),
            'selected': False,
            'visited_sites': [],
            'scout': True  # Bees start as scouts
        }
        bees.append(bee)

    visits = np.zeros((nestNum, num_bees))  # Tracks visits to each site per bee
    discoveries = np.zeros((nestNum, num_bees))  # Tracks discovery time for each site per bee
    current_time = np.zeros(num_bees)

    # Simulate bee behavior
    for bee_idx, bee in enumerate(bees):
        bee['path'].append(0)  # Start at the nest
        bee['times'].append(0)
        discoveries[0, bee_idx] = -1  # Discovered the nest
        visits[0, bee_idx] += 1
        explored_sites = {0}  # Nest is the initial explored site

        for visit in range(max_visits):
            # Determine next site
            if bee['scout']:
                # Scouts prioritize unexplored sites
                possible_sites = list(set(range(nestNum)) - explored_sites)
                if possible_sites:
                    new_site = np.random.choice(possible_sites)
                else:
                    bee['scout'] = False  # Switch to learned behavior
                    continue
            else:
                # Probabilistic selection based on learned probabilities
                probs = initial_probs[bee['path'][-1]]
                new_site = np.random.choice(range(nestNum), p=probs)

            # Update time and visits
            delta_time = distances[bee['path'][-1], new_site]
            current_time[bee_idx] += delta_time
            visits[new_site, bee_idx] += 1

            # Update path and learning
            bee['path'].append(new_site)
            bee['times'].append(current_time[bee_idx])
            explored_sites.add(new_site)

            if discoveries[new_site, bee_idx] == 0:
                discoveries[new_site, bee_idx] = current_time[bee_idx]

            # Reinforce probabilities based on visit count (decay factor)
            if decay_factor:
                initial_probs[:, new_site] *= decay_factor
                initial_probs[:, new_site] = np.clip(initial_probs[:, new_site], 0.01, None)
                initial_probs /= initial_probs.sum(axis=1, keepdims=True)

            # Termination criteria (optional)
            if len(explored_sites) == nestNum:
                break

    return current_time, discoveries, visits, bees