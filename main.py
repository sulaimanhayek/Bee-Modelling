import numpy as np
import matplotlib.pyplot as plt
from export_results_to_excel import export_results_to_excel

# Define flower positions (x, y coordinates), including home at (0, 0)
flower_positions = np.array([
    [0, 0],    # Home nest
    [1, 2],    # Flower 1
    [3, 4],    # Flower 2
    [5, 6],    # Flower 3
    [7, 1],    # Flower 4
    [4, 8]     # Flower 5
])

# Calculate distances between all points
num_flowers = len(flower_positions)
distances = np.linalg.norm(flower_positions[:, None] - flower_positions, axis=2)

# Initial probabilities based on distance
initial_probs = np.zeros((num_flowers, num_flowers))
for i in range(num_flowers):
    sorted_indices = np.argsort(distances[i])  # Sort by distance
    initial_probs[i, sorted_indices[1]] = 0.8  # Nearest flower (exclude self)
    initial_probs[i, sorted_indices[-1]] = 0.2  # Furthest flower
    initial_probs[i, :] /= initial_probs[i, :].sum()  # Normalize probabilities

# Transition probabilities for further visits
def update_probabilities(current_flower, visit_counts):
    probs = np.zeros(num_flowers)
    for i in range(num_flowers):
        if i == current_flower:
            continue
        if visit_counts[i] == 0:  # Prioritize unvisited flowers
            probs[i] = 0.6
        else:  # Other flowers or home
            probs[i] = 0.1
    return probs / probs.sum()  # Normalize

# Simulate bee visits
def simulate_bees(num_bees, max_visits):
    bee_paths = []
    total_distances = []

    for bee in range(num_bees):
        current_flower = 0  # Start at home nest
        path = [current_flower]
        visit_counts = np.zeros(num_flowers, dtype=int)
        visit_counts[current_flower] += 1
        total_distance = 0

        for _ in range(max_visits):
            # Choose next flower based on probabilities
            if sum(visit_counts) == 1:  # Initial visit
                probs = initial_probs[current_flower]
            else:
                probs = update_probabilities(current_flower, visit_counts)
            
            next_flower = np.random.choice(range(num_flowers), p=probs)
            path.append(next_flower)
            visit_counts[next_flower] += 1

            # Update total distance
            total_distance += distances[current_flower, next_flower]
            current_flower = next_flower

        bee_paths.append(path)
        total_distances.append(total_distance)

    return bee_paths, total_distances

# Visualize bee paths
def plot_bee_paths(bee_paths):
    plt.figure(figsize=(12, 8))
    for i, path in enumerate(bee_paths):
        plt.plot(path, marker='o', label=f'Bee {i+1}')
    plt.xlabel('Visit Number')
    plt.ylabel('Flower ID')
    plt.title('Bee Paths Across Flowers')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run simulation
num_bees = 7
max_visits = 30
bee_paths, total_distances = simulate_bees(num_bees, max_visits)

# Export results to Excel
export_results_to_excel(bee_paths, total_distances)

# Display results
for i, (path, distance) in enumerate(zip(bee_paths, total_distances)):
    print(f"Bee {i+1}: Path: {path}, Total Distance: {distance:.2f}")

# Plot paths
plot_bee_paths(bee_paths)