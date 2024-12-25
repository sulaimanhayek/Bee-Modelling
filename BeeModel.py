# import random
# import math
# import numpy as np

# def BeeModel(
#     verbose=True
# ):
#     """
#     A simulation where:
#       1) Flowers are arranged in a pentagon.
#       2) The nest (home) is placed outside the pentagon, equidistant from flower 1 and flower 5.
#       3) The bee starts at the nest, visits all flowers at least once, and returns to the nest.
#     """
#     # Pentagon configuration
#     radius = 5  # Distance from the pentagon center to each flower
#     angles = np.linspace(0, 2 * np.pi, 6)[:-1]  # Divide 360° into 5 points
#     flower_positions = [
#         (radius * math.cos(angle), radius * math.sin(angle)) for angle in angles
#     ]

#     # Calculate the nest position outside the pentagon
#     # We'll extend the line through flower 1 (first flower) and flower 5 (last flower)
#     flower_1 = flower_positions[0]
#     flower_5 = flower_positions[4]

#     # Midpoint between flower 1 and flower 5
#     midpoint = ((flower_1[0] + flower_5[0]) / 2, (flower_1[1] + flower_5[1]) / 2)

#     # Nest position further along the line (extending away from the pentagon center)
#     extension_factor = 1.5  # Determines how far the nest is from the midpoint
#     nest_position = (
#         midpoint[0] * extension_factor,
#         midpoint[1] * extension_factor,
#     )

#     path = [nest_position]
#     visited_indices = set()

#     # Randomly pick flowers until all 5 have been visited
#     while len(visited_indices) < 5:
#         next_index = random.randint(0, 4)
#         path.append(flower_positions[next_index])
#         visited_indices.add(next_index)

#     # Return to the nest
#     path.append(nest_position)

#     # Calculate distances
#     distances = []
#     total_distance = 0.0
#     for i in range(len(path) - 1):
#         dist = math.dist(path[i], path[i+1])
#         distances.append(dist)
#         total_distance += dist

#     results = {
#         "path": path,
#         "distances": distances,
#         "total_distance": total_distance,
#         "visited_flower_count": len(visited_indices),
#         "flower_positions": flower_positions,   # <--- For plotting
#         "nest_position": nest_position          # <--- For plotting
#     }

#     if verbose:
#         print("----- Bee Simulation Results -----")
#         print("Path:", path)
#         print("Stepwise Distances:", distances)
#         print(f"Total Distance: {total_distance:.2f}")
#         print("Visited Flower Count:", len(visited_indices))

#     return results

# if __name__ == "__main__":
#     output = BeeModel(verbose=True)

## random visits only


# import random
# import math

# def BeeModel(verbose=True):
#     """
#     A simulation where:
#       1) The nest is at (0, 0).
#       2) Flowers are positioned based on user-provided coordinates.
#       3) The bee starts at the nest, visits all flowers at least once, and returns.
#     """
#     # Define the nest and flower positions
#     nest_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4)    # Flower 5
#     ]

#     path = [nest_position]
#     visited_indices = set()

#     # Randomly pick flowers until all 5 have been visited
#     while len(visited_indices) < 5:
#         next_index = random.randint(0, 4)
#         path.append(flower_positions[next_index])
#         visited_indices.add(next_index)

#     # Return to the nest
#     path.append(nest_position)

#     # Calculate distances
#     distances = []
#     total_distance = 0.0
#     for i in range(len(path) - 1):
#         dist = math.dist(path[i], path[i+1])
#         distances.append(dist)
#         total_distance += dist

#     results = {
#         "path": path,
#         "distances": distances,
#         "total_distance": total_distance,
#         "visited_flower_count": len(visited_indices),
#         "flower_positions": flower_positions,
#         "nest_position": nest_position
#     }

#     if verbose:
#         print("----- Bee Simulation Results -----")
#         print("Path:", path)
#         print("Stepwise Distances:", distances)
#         print(f"Total Distance: {total_distance:.2f}")
#         print("Visited Flower Count:", len(visited_indices))

#     return results

# if __name__ == "__main__":
#     output = BeeModel(verbose=True)



## with learning mechanism


# import random
# import math

# def BeeModel(verbose=True):
#     """
#     A simulation where:
#       1) The bee starts at the nest and explores flowers randomly.
#       2) Uses reinforcement to prioritize shorter distances.
#       3) Ensures no flower is revisited until all flowers are visited.
#       4) Returns to the nest after visiting all flowers.
#     """
#     # Define the nest and flower positions
#     nest_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4)    # Flower 5
#     ]
#     num_flowers = len(flower_positions)

#     # Calculate distances between all flowers and the nest
#     all_positions = [nest_position] + flower_positions
#     distances = [
#         [math.dist(all_positions[i], all_positions[j]) for j in range(len(all_positions))]
#         for i in range(len(all_positions))
#     ]

#     # Path tracking
#     path = [0]  # Start at the nest (index 0 in all_positions)
#     visited = set()

#     # Reinforcement factor
#     reinforcement_factor = 2

#     def calculate_probabilities(current_index):
#         """
#         Calculate the probabilities of visiting each flower based on distances
#         and reinforcement learning.
#         """
#         probs = [0] * (num_flowers + 1)  # Include the nest and flowers
#         for i in range(1, num_flowers + 1):  # Flowers only (indices 1 to num_flowers)
#             if i not in visited:
#                 if distances[current_index][i] == min(distances[current_index][j] for j in range(1, num_flowers + 1) if j not in visited):
#                     # Nearest flower
#                     probs[i] = 0.8 * reinforcement_factor
#                 else:
#                     # Other flowers
#                     probs[i] = 0.2 * reinforcement_factor
#         # Normalize probabilities
#         total = sum(probs)
#         if total > 1:
#             probs = [p / total for p in probs]
#         return probs

#     # Main simulation loop
#     while len(visited) < num_flowers:
#         current_index = path[-1]
#         probabilities = calculate_probabilities(current_index)
#         next_index = random.choices(range(len(all_positions)), probabilities)[0]
#         path.append(next_index)
#         visited.add(next_index)

#     # Return to the nest
#     path.append(0)

#     # Calculate total distance traveled
#     total_distance = 0.0
#     step_distances = []
#     for i in range(len(path) - 1):
#         dist = distances[path[i]][path[i + 1]]
#         step_distances.append(dist)
#         total_distance += dist

#     # Convert path indices to coordinates
#     coordinate_path = [all_positions[i] for i in path]

#     results = {
#         "path": coordinate_path,
#         "distances": step_distances,
#         "total_distance": total_distance,
#         "visited_flower_count": len(visited),
#         "flower_positions": flower_positions,
#         "nest_position": nest_position
#     }

#     if verbose:
#         print("----- Bee Simulation Results -----")
#         print("Path:", coordinate_path)
#         print("Stepwise Distances:", step_distances)
#         print(f"Total Distance: {total_distance:.2f}")
#         print("Visited Flower Count:", len(visited))

#     return results

# if __name__ == "__main__":
#     output = BeeModel(verbose=True)





# import random
# import math

# def BeeModel(verbose=True):
#     """
#     A simulation where:
#       1) The bee performs a scouting trip to collect distances.
#       2) Decision-making uses probabilities weighted by distances and reinforcement.
#       3) The bee avoids revisiting flowers until all are visited, then returns to the nest.
#     """
#     # Define the nest and flower positions
#     nest_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4)    # Flower 5
#     ]
#     num_flowers = len(flower_positions)

#     # Combine all positions (nest + flowers) for indexing
#     all_positions = [nest_position] + flower_positions

#     # Calculate distances between all points
#     distances = [
#         [math.dist(all_positions[i], all_positions[j]) for j in range(len(all_positions))]
#         for i in range(len(all_positions))
#     ]

#     # Scouting phase: Randomly visit all flowers
#     visited = set()
#     random_path = [0]  # Start at the nest (index 0)
#     while len(visited) < num_flowers:
#         next_index = random.randint(1, num_flowers)  # Randomly pick flowers (1 to num_flowers)
#         if next_index not in visited:
#             random_path.append(next_index)
#             visited.add(next_index)
#     random_path.append(0)  # Return to the nest
#     if verbose:
#         print("Scouting Path:", [all_positions[i] for i in random_path])

#     # Decision-making phase
#     decision_path = [0]  # Start at the nest
#     visited = set()
#     reinforcement_factor = 2.0

#     def calculate_probabilities(current_index, unvisited):
#         """
#         Calculate probabilities based on distances and reinforcement factor.
#         """
#         probs = [0] * (num_flowers + 1)  # Include nest and flowers
#         for i in unvisited:
#             if distances[current_index][i] == min(distances[current_index][j] for j in unvisited):
#                 probs[i] = 0.8 * reinforcement_factor  # Nearest flower
#             else:
#                 probs[i] = 0.2 / (len(unvisited) - 1) * reinforcement_factor  # Other flowers
#         # Normalize probabilities
#         total = sum(probs)
#         probs = [p / total if total > 0 else 0 for p in probs]
#         return probs

#     while len(visited) < num_flowers:
#         current_index = decision_path[-1]
#         unvisited = {i for i in range(1, num_flowers + 1) if i not in visited}
#         probabilities = calculate_probabilities(current_index, unvisited)
#         next_index = random.choices(range(len(all_positions)), probabilities)[0]
#         decision_path.append(next_index)
#         visited.add(next_index)
#         reinforcement_factor *= 2  # Apply reinforcement factor after each move

#     # Return to the nest
#     decision_path.append(0)

#     # Calculate total distance traveled in the decision-making phase
#     total_distance = 0.0
#     step_distances = []
#     for i in range(len(decision_path) - 1):
#         dist = distances[decision_path[i]][decision_path[i + 1]]
#         step_distances.append(dist)
#         total_distance += dist

#     # Convert path indices to coordinates
#     decision_coordinates = [all_positions[i] for i in decision_path]

#     results = {
#         "scouting_path": [all_positions[i] for i in random_path],
#         "decision_path": decision_coordinates,
#         "distances": step_distances,
#         "total_distance": total_distance,
#         "flower_positions": flower_positions,
#         "nest_position": nest_position
#     }

#     if verbose:
#         print("----- Bee Simulation Results -----")
#         print("Scouting Path:", results["scouting_path"])
#         print("Decision Path:", results["decision_path"])
#         print("Stepwise Distances:", step_distances)
#         print(f"Total Distance: {total_distance:.2f}")

#     return results

# if __name__ == "__main__":
#     output = BeeModel(verbose=True)





# with distance integration 


# import math

# def BeeModel(verbose=True):
#     """
#     A simulation where:
#       1) The bee starts at the nest.
#       2) Flowers are positioned at predefined coordinates.
#       3) The bee deterministically moves to the nearest unvisited flower, 
#          based on a stored distance lookup.
#       4) Once all flowers are visited, the bee returns to the nest.
#     """
#     # -------------------------
#     # 1. Define all coordinates
#     # -------------------------
#     # In this example, index 0 = nest, indices 1..N = flowers
#     nest_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4)    # Flower 5
#     ]

#     all_positions = [nest_position] + flower_positions  # Combine nest + flowers
#     num_flowers = len(flower_positions)

#     # -------------------------
#     # 2. Build the distance map
#     # -------------------------
#     # distance_lookup[(i, j)] = distance from point i to point j
#     distance_lookup = {}
#     for i in range(len(all_positions)):
#         for j in range(len(all_positions)):
#             distance_lookup[(i, j)] = math.dist(all_positions[i], all_positions[j])

#     # -------------------------
#     # 3. Deterministic pathing
#     # -------------------------
#     path = [0]        # Start at the nest (index 0)
#     visited = set()   # Track visited flowers (1..N)

#     while len(visited) < num_flowers:
#         current_index = path[-1]
#         # Figure out which flowers are still unvisited
#         unvisited = [f for f in range(1, num_flowers + 1) if f not in visited]

#         # Pick the nearest unvisited flower from current position
#         next_index = min(unvisited, key=lambda x: distance_lookup[(current_index, x)])
        
#         # Move there
#         visited.add(next_index)
#         path.append(next_index)

#     # After visiting all flowers, return to the nest (index 0)
#     path.append(0)

#     # -------------------------
#     # 4. Compute stepwise distances
#     # -------------------------
#     step_distances = []
#     total_distance = 0.0
#     for i in range(len(path) - 1):
#         dist = distance_lookup[(path[i], path[i+1])]
#         step_distances.append(dist)
#         total_distance += dist

#     # Convert path indices to coordinates for readability
#     coordinate_path = [all_positions[i] for i in path]

#     # -------------------------
#     # 5. Summarize results
#     # -------------------------
#     results = {
#         "path_indices": path,
#         "path_coordinates": coordinate_path,
#         "step_distances": step_distances,
#         "total_distance": total_distance,
#         "visited_flowers": len(visited),
#         "flower_positions": flower_positions,
#         "nest_position": nest_position,
#         "distance_lookup_example": {
#             # For illustration, let's show a few entries
#             "(0 -> 1)": distance_lookup[(0,1)],
#             "(1 -> 2)": distance_lookup[(1,2)],
#             "(2 -> 3)": distance_lookup[(2,3)]
#         }
#     }

#     # -------------------------
#     # 6. Optionally, print out
#     # -------------------------
#     if verbose:
#         print("----- Bee Simulation Results (Deterministic) -----")
#         print("Path (indices):       ", path)
#         print("Path (coordinates):   ", coordinate_path)
#         print("Stepwise distances:   ", step_distances)
#         print(f"Total distance:       {total_distance:.2f}")
#         print("Visited flower count: ", results["visited_flowers"])
#         print("Sample distance lookups:")
#         for k, v in results["distance_lookup_example"].items():
#             print(f"  {k}: {v:.2f}")

#     return results

# if __name__ == "__main__":
#     BeeModel(verbose=True)



# import numpy as np

# class BeeModel:
#     def __init__(self, current_position, points):
#         """
#         Initializes the BeeModel.
        
#         :param current_position: A tuple representing the bee's current position (x, y).
#         :param points: A list of tuples representing the other points [(x1, y1), (x2, y2), ...].
#         """
#         self.current_position = current_position
#         self.points = points

#     def calculate_distances(self):
#         """
#         Calculate distances from the current position to all other points.
        
#         :return: A list of tuples (distance, point), sorted by distance (shortest to furthest).
#         """
#         distances = []
#         for point in self.points:
#             distance = np.linalg.norm(np.array(self.current_position) - np.array(point))
#             distances.append((distance, point))
#         return sorted(distances, key=lambda x: x[0])

#     def calculate_normalized_weights(self, sorted_distances):
#         """
#         Calculate normalized weights for all points based on distance.
        
#         :param sorted_distances: A list of tuples (distance, point), sorted by distance.
#         :return: A dictionary of points and their normalized probabilities.
#         """
#         weights = []
#         for i, (distance, _) in enumerate(sorted_distances):
#             if i == 0:  # Closest point
#                 weight = distance * 0.8
#             else:
#                 weight = distance * 0.2
#             weights.append(weight)

#         # Normalize weights
#         total_weight = sum(weights)
#         probabilities = [weight / total_weight for weight in weights]

#         # Create a dictionary mapping points to their probabilities
#         points = [point for _, point in sorted_distances]
#         return dict(zip(points, probabilities))

#     def move_to_next_point(self):
#         """
#         Determine the next point to move to based on normalized probabilities.
        
#         :return: The next point the bee will move to.
#         """
#         sorted_distances = self.calculate_distances()
#         normalized_weights = self.calculate_normalized_weights(sorted_distances)
#         # Select the point with the highest probability
#         next_point = max(normalized_weights, key=normalized_weights.get)
#         self.current_position = next_point  # Update current position
#         return next_point


# # Example usage
# if __name__ == "__main__":
#     # Example data
#     current_position = (0, 0)
#     points = [
#     (-4, -4),  # Flower 1
#     (-4, 12),  # Flower 2
#     (0, 20),   # Flower 3
#     (4, 12),   # Flower 4
#     (4, -4)]    # Flower 5

#     # Initialize the BeeModel
#     bee = BeeModel(current_position, points)

#     # Calculate sorted distances
#     sorted_distances = bee.calculate_distances()
#     print("Sorted Distances:", sorted_distances)

#     # Calculate normalized weights
#     normalized_weights = bee.calculate_normalized_weights(sorted_distances)
#     print("Normalized Weights:", normalized_weights)

#     # Determine the next point to move to
#     next_point = bee.move_to_next_point()
#     print("Next Point:", next_point)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# class BeeModel:
#     def __init__(self, start_position, flower_positions, reinforcement_factor=2.0, max_round_trips=30):
#         self.start_position = start_position
#         self.flower_positions = flower_positions
#         self.num_flowers = len(flower_positions)
#         self.reinforcement_factor = reinforcement_factor
#         self.max_round_trips = max_round_trips
#         self.distances = self._calculate_distance_matrix()
#         self.probabilities = [0.6] + [0.1] * (self.num_flowers - 1)  # Initial probabilities
#         self.round_trip_data = []

#     def _calculate_distance_matrix(self):
#         all_positions = [self.start_position] + self.flower_positions
#         distances = [[np.linalg.norm(np.array(p1) - np.array(p2)) for p2 in all_positions] for p1 in all_positions]
#         return distances

#     def _choose_next_flower(self, current_index, visited):
#         unvisited = [i for i in range(1, self.num_flowers + 1) if i not in visited]
#         if not unvisited:
#             return 0  # All flowers visited, return to the nest

#         probabilities = [self.probabilities[i - 1] for i in unvisited]
#         total_prob = sum(probabilities)
#         probabilities = [p / total_prob for p in probabilities]

#         return np.random.choice(unvisited, p=probabilities)

#     def _update_probabilities_with_memory(self, path, total_distance):
#         path_distances = [self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1)]
#         distance_weighted_probs = [1 / d if d > 0 else 0 for d in path_distances]
#         reinforced_probs = [p * self.reinforcement_factor for p in distance_weighted_probs]
#         total_prob = sum(reinforced_probs)
#         self.probabilities = [p / total_prob for p in reinforced_probs]

#     def _plot_paths(self, all_positions, paths, total_distances):
#         # Plot each round trip
#         for i, path in enumerate(paths):
#             plt.figure(figsize=(6, 6))
#             x, y = zip(*[all_positions[idx] for idx in path])
#             plt.plot(x, y, marker="o", label=f"Round Trip {i + 1}")
#             plt.scatter(*zip(*all_positions), color="red", label="Flowers", zorder=5)
#             plt.scatter(*self.start_position, color="blue", label="Nest", zorder=5)
#             plt.title(f"Bee Path - Round Trip {i + 1}")
#             plt.xlabel("X Coordinate")
#             plt.ylabel("Y Coordinate")
#             plt.legend()
#             plt.grid()
#             plt.show()

#         # Plot total distance across round trips
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, len(total_distances) + 1), total_distances, marker="o", linestyle="-")
#         plt.title("Total Distance Across Round Trips")
#         plt.xlabel("Round Trip")
#         plt.ylabel("Total Distance")
#         plt.grid()
#         plt.show()

#     def simulate(self):
#         all_positions = [self.start_position] + self.flower_positions
#         paths = []  # To store paths for visualization
#         total_distances = []  # To store total distances for visualization

#         for round_trip in range(self.max_round_trips):
#             path = [0]  # Start at the nest
#             visited = set()

#             while len(visited) < self.num_flowers:
#                 current_index = path[-1]
#                 next_index = self._choose_next_flower(current_index, visited)
#                 path.append(next_index)
#                 visited.add(next_index)

#             path.append(0)  # Return to the nest
#             paths.append(path)

#             # Calculate total distance for the round trip
#             total_distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
#             total_distances.append(total_distance)

#             # Record the round trip data
#             self.round_trip_data.append({
#                 "Round Trip": round_trip + 1,
#                 "Path": [all_positions[i] for i in path],
#                 "Total Distance": total_distance,
#                 "Probabilities": self.probabilities.copy()
#             })

#             # Update probabilities using internal memory and reinforcement
#             self._update_probabilities_with_memory(path, total_distance)

#         # Visualize paths and total distances
#         self._plot_paths(all_positions, paths, total_distances)
#         return self.round_trip_data

#     def export_to_excel(self, filename="bee_simulation_results_with_memory.xlsx"):
#         data = []
#         for trip in self.round_trip_data:
#             for i, prob in enumerate(trip["Probabilities"]):
#                 data.append({
#                     "Round Trip": trip["Round Trip"],
#                     "Flower": i + 1,
#                     "Probability": prob,
#                     "Path": " -> ".join(map(str, trip["Path"])),
#                     "Total Distance": trip["Total Distance"]
#                 })

#         df = pd.DataFrame(data)
#         df.to_excel(filename, index=False)

# # Define the nest and flower positions
# start_position = (0, -12)
# flower_positions = [
#     (-4, -4),  # Flower 1
#     (-4, 12),  # Flower 2
#     (0, 20),   # Flower 3
#     (4, 12),   # Flower 4
#     (4, -4)    # Flower 5
# ]

# # Initialize and simulate
# bee = BeeModel(start_position, flower_positions)
# results = bee.simulate()
# bee.export_to_excel("bee_simulation_results_with_memory.xlsx")

# print("Simulation completed. Results exported to 'bee_simulation_results_with_memory.xlsx'.")


## With TSP combination


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import permutations

# class BeeModel:
#     def __init__(self, start_position, flower_positions, reinforcement_factor=2.0, max_round_trips=30):
#         self.start_position = start_position
#         self.flower_positions = flower_positions
#         self.num_flowers = len(flower_positions)
#         self.reinforcement_factor = reinforcement_factor
#         self.max_round_trips = max_round_trips
#         self.distances = self._calculate_distance_matrix()
#         self.probabilities = [0.6] + [0.1] * (self.num_flowers - 1)  # Initial probabilities
#         self.round_trip_data = []

#     def _calculate_distance_matrix(self):
#         all_positions = [self.start_position] + self.flower_positions
#         distances = [[np.linalg.norm(np.array(p1) - np.array(p2)) for p2 in all_positions] for p1 in all_positions]
#         return distances

#     def _solve_tsp(self):
#         """
#         Solve the Traveling Salesman Problem (TSP) using brute force for small problem size.
#         Returns the shortest path and its distance.
#         """
#         points = list(range(1, self.num_flowers + 1))  # Flowers only (indices 1 to N)
#         best_distance = float('inf')
#         best_path = []

#         for perm in permutations(points):
#             # Add nest at start and end of the path
#             full_path = [0] + list(perm) + [0]
#             distance = sum(self.distances[full_path[i]][full_path[i + 1]] for i in range(len(full_path) - 1))

#             if distance < best_distance:
#                 best_distance = distance
#                 best_path = full_path

#         return best_path, best_distance

#     def _choose_next_flower(self, current_index, visited):
#         unvisited = [i for i in range(1, self.num_flowers + 1) if i not in visited]
#         if not unvisited:
#             return 0  # All flowers visited, return to the nest

#         probabilities = [self.probabilities[i - 1] for i in unvisited]
#         total_prob = sum(probabilities)
#         probabilities = [p / total_prob for p in probabilities]

#         return np.random.choice(unvisited, p=probabilities)

#     def _update_probabilities_with_tsp(self, bee_path, tsp_path):
#         """
#         Update probabilities to align with the TSP-calculated shortest path.
#         """
#         # Reward moves that align with the TSP path
#         path_distances = [self.distances[bee_path[i]][bee_path[i + 1]] for i in range(len(bee_path) - 1)]
#         tsp_distances = [self.distances[tsp_path[i]][tsp_path[i + 1]] for i in range(len(tsp_path) - 1)]

#         # Reinforce probabilities for steps closer to the TSP solution
#         reinforced_probs = [0.0] * (self.num_flowers + 1)
#         for i in range(len(tsp_path) - 1):
#             reinforced_probs[tsp_path[i + 1] - 1] += self.reinforcement_factor / tsp_distances[i]

#         # Normalize probabilities
#         total_prob = sum(reinforced_probs)
#         self.probabilities = [p / total_prob for p in reinforced_probs]

#     def simulate(self):
#         all_positions = [self.start_position] + self.flower_positions
#         paths = []  # To store paths for visualization
#         total_distances = []  # To store total distances for visualization

#         # Compute the optimal TSP path before starting the simulation
#         tsp_path, tsp_distance = self._solve_tsp()
#         print(f"Optimal TSP Path: {[all_positions[i] for i in tsp_path]} with Distance: {tsp_distance:.2f}")

#         for round_trip in range(self.max_round_trips):
#             path = [0]  # Start at the nest
#             visited = set()

#             while len(visited) < self.num_flowers:
#                 current_index = path[-1]
#                 next_index = self._choose_next_flower(current_index, visited)
#                 path.append(next_index)
#                 visited.add(next_index)

#             path.append(0)  # Return to the nest
#             paths.append(path)

#             # Calculate total distance for the round trip
#             total_distance = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
#             total_distances.append(total_distance)

#             # Record the round trip data
#             self.round_trip_data.append({
#                 "Round Trip": round_trip + 1,
#                 "Path": [all_positions[i] for i in path],
#                 "Total Distance": total_distance,
#                 "Probabilities": self.probabilities.copy()
#             })

#             # Update probabilities using the TSP solution
#             self._update_probabilities_with_tsp(path, tsp_path)

#         # Visualize paths and total distances
#         self._plot_paths(all_positions, paths, total_distances)
#         return self.round_trip_data

#     def _plot_paths(self, all_positions, paths, total_distances):
#         # Plot each round trip
#         for i, path in enumerate(paths):
#             plt.figure(figsize=(6, 6))
#             x, y = zip(*[all_positions[idx] for idx in path])
#             plt.plot(x, y, marker="o", label=f"Round Trip {i + 1}")
#             plt.scatter(*zip(*all_positions), color="red", label="Flowers", zorder=5)
#             plt.scatter(*self.start_position, color="blue", label="Nest", zorder=5)
#             plt.title(f"Bee Path - Round Trip {i + 1}")
#             plt.xlabel("X Coordinate")
#             plt.ylabel("Y Coordinate")
#             plt.legend()
#             plt.grid()
#             plt.show()

#         # Plot total distance across round trips
#         plt.figure(figsize=(10, 6))
#         plt.plot(range(1, len(total_distances) + 1), total_distances, marker="o", linestyle="-")
#         plt.title("Total Distance Across Round Trips")
#         plt.xlabel("Round Trip")
#         plt.ylabel("Total Distance")
#         plt.grid()
#         plt.show()

#     def export_to_excel(self, filename="bee_simulation_results_with_tsp.xlsx"):
#         data = []
#         for trip in self.round_trip_data:
#             for i, prob in enumerate(trip["Probabilities"]):
#                 data.append({
#                     "Round Trip": trip["Round Trip"],
#                     "Flower": i + 1,
#                     "Probability": prob,
#                     "Path": " -> ".join(map(str, trip["Path"])),
#                     "Total Distance": trip["Total Distance"]
#                 })

#         df = pd.DataFrame(data)
#         df.to_excel(filename, index=False)

# # Define the nest and flower positions
# start_position = (0, -12)
# flower_positions = [
#     (-4, -4),  # Flower 1
#     (-4, 12),  # Flower 2
#     (0, 20),   # Flower 3
#     (4, 12),   # Flower 4
#     (4, -4),    # Flower 5
# #    (10,10)    # Additional flower
# ]

# # Initialize and simulate
# bee = BeeModel(start_position, flower_positions)
# results = bee.simulate()
# bee.export_to_excel("bee_simulation_results_with_tsp.xlsx")

# print("Simulation completed. Results exported to 'bee_simulation_results_with_tsp.xlsx'.")





## combination of TSP , Probabilities of 0.6 and 0.1 and reinforcement factor

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import permutations

# class BeeModel:
#     def __init__(self, start_position, flower_positions, reinforcement_factor=2.0, max_round_trips=30):
#         """
#         :param start_position: The coordinates of the nest, e.g. (x, y).
#         :param flower_positions: A list of (x, y) positions for flowers.
#         :param reinforcement_factor: Factor by which we reinforce TSP transitions when we find a better path.
#         :param max_round_trips: Number of round trips the bee will take.
#         """
#         self.start_position = start_position
#         self.flower_positions = flower_positions
#         self.num_flowers = len(flower_positions)
#         self.reinforcement_factor = reinforcement_factor
#         self.max_round_trips = max_round_trips
        
#         # Distance matrix: index 0 is the nest, indices [1..N] are flowers
#         self.distances = self._calculate_distance_matrix()  
        
#         # TSP-based transitions (for each pair i->j, track if it is on TSP path)
#         # We'll store and update probabilities of going from i to j over time.
#         # Initialize all with 0.1, except TSP edges with 0.6 (or distributed among TSP edges).
#         self.transition_probabilities = None
        
#         self.round_trip_data = []
#         self.best_distance = float("inf")  # Track the best (shortest) total distance seen so far

#     def _calculate_distance_matrix(self):
#         """Compute a 2D matrix of pairwise distances among nest + all flowers."""
#         all_positions = [self.start_position] + self.flower_positions
#         n = len(all_positions)
#         distances = [[0]*n for _ in range(n)]
#         for i in range(n):
#             for j in range(n):
#                 distances[i][j] = np.linalg.norm(np.array(all_positions[i]) - np.array(all_positions[j]))
#         return distances

#     def _solve_tsp(self):
#         """
#         Brute-force TSP solver for demonstration (works for small sets of flowers).
#         Returns:
#           tsp_path: the best path as a list of indices (0 is nest, 1..N are flowers),
#                     including 0 at the start and end.
#           tsp_distance: total distance of this best path.
#         """
#         points = list(range(1, self.num_flowers + 1))  # Flower indices only
#         best_distance = float("inf")
#         best_path = []

#         for perm in permutations(points):
#             path = [0] + list(perm) + [0]  # Start/end at nest
#             dist = sum(self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1))
#             if dist < best_distance:
#                 best_distance = dist
#                 best_path = path
        
#         return best_path, best_distance

#     def _init_transition_probabilities(self, tsp_path):
#         """
#         Initialize the transition probabilities for each pair (i -> j).
        
#         We'll create a matrix "transition_probabilities" of shape (N+1, N+1),
#         where N is the number of flowers, index 0 is nest.
        
#         If (i -> j) is on the TSP path, set base probability to 0.6.
#         For all other unvisited flowers, set base probability to 0.1.
        
#         Note: We must ensure row-wise normalization so that from any node i,
#         the sum of probabilities to unvisited j is 1.0 (or close to it).
#         """
#         n = self.num_flowers + 1  # 0..N
#         probs = [[0.0]*n for _ in range(n)]
        
#         # Mark edges that are on TSP path
#         tsp_edges = set()
#         for i in range(len(tsp_path) - 1):
#             tsp_edges.add((tsp_path[i], tsp_path[i + 1]))

#         # Assign base probabilities
#         for i in range(n):
#             # Find how many edges from i are TSP edges (to distribute 0.6)
#             # and how many are non-TSP (distribute 0.1).
#             # But note: we might not want them to sum exactly to 1 across TSP vs non-TSP,
#             # because we only want to define relative preference. We'll do a row normalization after.
#             for j in range(n):
#                 if i != j:
#                     if (i, j) in tsp_edges:
#                         # If i->j is on TSP path, start with higher prob
#                         probs[i][j] = 0.6
#                     else:
#                         # Non-TSP transitions start lower
#                         probs[i][j] = 0.1
#                 else:
#                     # No self-transitions
#                     probs[i][j] = 0.0

#             # Normalize row i so that sum(probs[i]) = 1.0
#             row_sum = sum(probs[i])
#             if row_sum > 0:
#                 probs[i] = [p / row_sum for p in probs[i]]
        
#         self.transition_probabilities = probs

#     def _reinforce_tsp_edges(self, path):
#         """
#         If this path is better than the best known distance, reinforce TSP edges
#         by multiplying them by reinforcement_factor, and then re-normalize.
        
#         path: The actual round-trip path taken by the bee (list of indices).
#         """
#         # Compare distance with best_distance
#         distance = self._compute_path_distance(path)
#         if distance < self.best_distance:
#             # We found a better path => reinforce TSP edges within the adjacency matrix
#             n = self.num_flowers + 1
#             for i in range(n):
#                 # Multiply all TSP edges (in row i) by the reinforcement factor
#                 # Actually, we only want to do this for edges that have a higher baseline
#                 # but in this simplified approach, let's multiply all existing high edges
#                 # or we can multiply entire row i to keep it simple. 
#                 for j in range(n):
#                     if i != j and self.transition_probabilities[i][j] > 0.5:  # threshold for "TSP-like"
#                         self.transition_probabilities[i][j] *= self.reinforcement_factor
                        
#             # Save new best distance
#             self.best_distance = distance

#         # Re-normalize each row so it sums to 1
#         self._normalize_transition_probabilities()

#     def _normalize_transition_probabilities(self):
#         """Normalize each row in transition_probabilities so that it sums to 1.0."""
#         n = self.num_flowers + 1
#         for i in range(n):
#             row_sum = sum(self.transition_probabilities[i])
#             if row_sum > 0:
#                 self.transition_probabilities[i] = [
#                     p / row_sum for p in self.transition_probabilities[i]
#                 ]

#     def _choose_next_flower(self, current_index, visited):
#         """
#         Pick the next flower using the row of transition_probabilities[current_index],
#         but only among unvisited flowers + possibly the nest if everything is visited.
        
#         :param current_index: index of the current position (0=nest, 1..N=flowers).
#         :param visited: set of visited flower indices (1..N).
#         :return: index of the chosen next flower (or 0 if done).
#         """
#         unvisited = [f for f in range(1, self.num_flowers + 1) if f not in visited]
        
#         # If everything is visited, we want to go back to nest
#         if not unvisited:
#             return 0
        
#         # Convert the row of transition probabilities into a distribution
#         # but only for unvisited. We'll do a quick mask & re-normalize.
#         row_probs = self.transition_probabilities[current_index]
#         # For j in unvisited, we keep row_probs[j], for the rest we set 0
#         masked_probs = [row_probs[j] if j in unvisited else 0 for j in range(len(row_probs))]
        
#         # If the sum is 0 (which can happen if we are on some row that had no TSP edges),
#         # default to uniform among unvisited
#         s = sum(masked_probs)
#         if s <= 1e-9:
#             # fallback uniform distribution among unvisited
#             uniform = 1.0 / len(unvisited)
#             for j in range(len(masked_probs)):
#                 if j in unvisited:
#                     masked_probs[j] = uniform
#                 else:
#                     masked_probs[j] = 0.0
#         else:
#             # Re-normalize
#             masked_probs = [p / s for p in masked_probs]

#         # Choose next index from masked_probs
#         next_index = np.random.choice(range(len(masked_probs)), p=masked_probs)
#         return next_index

#     def _compute_path_distance(self, path):
#         """Sum up the distances along the path indices."""
#         return sum(self.distances[path[i]][path[i+1]] for i in range(len(path)-1))

#     def simulate(self):
#         """
#         Run the simulation for max_round_trips:
#          1. Solve TSP to get best path (tsp_path).
#          2. Initialize transition probabilities using TSP edges = 0.6, others = 0.1.
#          3. Each round trip:
#              - Bee moves until all flowers are visited, then returns to nest.
#              - Calculate total distance.
#              - Compare with best distance & reinforce TSP edges if improved.
#         """
#         all_positions = [self.start_position] + self.flower_positions
        
#         # 1. Solve TSP once
#         tsp_path, tsp_distance = self._solve_tsp()
#         print(f"[TSP] Optimal path: {[all_positions[i] for i in tsp_path]}, Distance = {tsp_distance:.2f}")
        
#         # 2. Initialize transition probabilities
#         self._init_transition_probabilities(tsp_path)
        
#         # 3. Simulate multiple round trips
#         paths_for_plot = []
#         distances_for_plot = []

#         for round_idx in range(self.max_round_trips):
#             # Perform one round trip
#             path = [0]     # start at nest
#             visited = set()
            
#             while len(visited) < self.num_flowers:
#                 current_index = path[-1]
#                 next_index = self._choose_next_flower(current_index, visited)
#                 path.append(next_index)
#                 visited.add(next_index)
            
#             # Return to nest
#             path.append(0)
#             paths_for_plot.append(path)

#             # Compute distance
#             total_distance = self._compute_path_distance(path)
#             distances_for_plot.append(total_distance)

#             # Store data
#             self.round_trip_data.append({
#                 "Round Trip": round_idx + 1,
#                 "Path Indices": path,
#                 "Path Coordinates": [all_positions[i] for i in path],
#                 "Total Distance": total_distance
#             })

#             # Compare & possibly reinforce TSP edges
#             self._reinforce_tsp_edges(path)

#         # 4. Visualize
#         self._plot_paths(all_positions, paths_for_plot, distances_for_plot)
#         return self.round_trip_data

#     def _plot_paths(self, all_positions, paths, distances):
#         """Plot each round trip path and the distance evolution over time."""
#         # Plot each round trip path
#         for i, path in enumerate(paths):
#             plt.figure(figsize=(6, 6))
#             xx = [all_positions[idx][0] for idx in path]
#             yy = [all_positions[idx][1] for idx in path]
#             plt.plot(xx, yy, marker="o", label=f"Round Trip {i + 1}")
            
#             # Mark the nest in blue and flowers in red
#             plt.scatter(all_positions[0][0], all_positions[0][1], c="blue", s=80, label="Nest")
#             plt.scatter(
#                 [p[0] for p in all_positions[1:]],
#                 [p[1] for p in all_positions[1:]],
#                 c="red", s=80, label="Flowers"
#             )
#             plt.title(f"Bee Path - Round Trip {i + 1}")
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.legend()
#             plt.grid()
#             plt.show()
        
#         # Plot total distance across round trips
#         plt.figure(figsize=(8, 5))
#         plt.plot(range(1, len(distances) + 1), distances, marker="o")
#         plt.title("Total Distance Over Round Trips")
#         plt.xlabel("Round Trip")
#         plt.ylabel("Total Distance")
#         plt.grid()
#         plt.show()

#     def export_to_excel(self, filename="bee_simulation_results_with_tsp_and_prob.xlsx"):
#         """
#         Export round trip data to Excel, including path and total distance.
#         """
#         records = []
#         for trip in self.round_trip_data:
#             path_str = " -> ".join(map(str, trip["Path Indices"]))
#             records.append({
#                 "Round Trip": trip["Round Trip"],
#                 "Path Indices": path_str,
#                 "Path Coordinates": trip["Path Coordinates"],
#                 "Total Distance": trip["Total Distance"]
#             })
#         df = pd.DataFrame(records)
#         df.to_excel(filename, index=False)
#         print(f"Results exported to {filename}")


# # -------------
# # USAGE EXAMPLE
# # -------------

# if __name__ == "__main__":
#     # 1. Define nest (0) and flowers (1..N)
#     start_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4),   # Flower 5
#         # Add more flowers if desired
#     ]
    
#     # 2. Create and run the BeeModel
#     bee_model = BeeModel(
#         start_position=start_position,
#         flower_positions=flower_positions,
#         reinforcement_factor=2.0,
#         max_round_trips=30
#     )
    
#     # 3. Simulate
#     results = bee_model.simulate()
    
#     # 4. Export to Excel
#     bee_model.export_to_excel("bee_simulation_results_with_tsp_and_prob.xlsx")

#     print("Simulation complete.")


## combination of TSP , Probabilities of 0.6 and 0.1 and reinforcement factor,
## with multiple visits



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import permutations

# class BeeModel:
#     def __init__(
#         self,
#         start_position,
#         flower_positions,
#         reinforcement_factor=2.0,
#         max_round_trips=30,
#         steps_per_trip=None
#     ):
#         """
#         :param start_position: The coordinates of the nest, e.g. (x, y).
#         :param flower_positions: A list of (x, y) positions for flowers.
#         :param reinforcement_factor: Factor by which we reinforce TSP transitions
#                                      when we find a better path.
#         :param max_round_trips: Number of round trips the bee will take.
#         :param steps_per_trip: How many 'moves' the bee makes before returning to the nest.
#                                If None, defaults to number_of_flowers.
#         """
#         self.start_position = start_position
#         self.flower_positions = flower_positions
#         self.num_flowers = len(flower_positions)
#         self.reinforcement_factor = reinforcement_factor
#         self.max_round_trips = max_round_trips
        
#         # If user doesn't specify steps per trip, default to #flowers
#         self.steps_per_trip = steps_per_trip or self.num_flowers
        
#         # 0 = nest, 1..N = flowers
#         self.distances = self._calculate_distance_matrix()  
        
#         # Will hold transition probabilities: (N+1) x (N+1)
#         # Initialized after TSP is solved
#         self.transition_probabilities = None
        
#         # Data logging
#         self.round_trip_data = []
#         self.best_distance = float("inf")  # Track shortest total distance so far

#     def _calculate_distance_matrix(self):
#         """Compute a 2D matrix of pairwise distances among nest + all flowers."""
#         all_positions = [self.start_position] + self.flower_positions
#         n = len(all_positions)
#         dist_matrix = [[0.0]*n for _ in range(n)]
#         for i in range(n):
#             for j in range(n):
#                 dist_matrix[i][j] = np.linalg.norm(
#                     np.array(all_positions[i]) - np.array(all_positions[j])
#                 )
#         return dist_matrix

#     def _solve_tsp(self):
#         """
#         Brute-force TSP solver (works for small #flowers).
#         Returns:
#           tsp_path: list of indices (0 = nest, 1..N = flowers) including 0 at start & end.
#           tsp_distance: total distance of that route.
#         """
#         points = list(range(1, self.num_flowers + 1))
#         best_distance = float("inf")
#         best_path = []
#         for perm in permutations(points):
#             path = [0] + list(perm) + [0]
#             dist = sum(self.distances[path[i]][path[i+1]] for i in range(len(path) - 1))
#             if dist < best_distance:
#                 best_distance = dist
#                 best_path = path
#         return best_path, best_distance

#     def _init_transition_probabilities(self, tsp_path):
#         """
#         Initialize transition probabilities matrix (N+1 x N+1).
#         If (i->j) is on the TSP path, give base prob = 0.6.
#         Otherwise, base prob = 0.1.
#         Then row-normalize.
#         """
#         n = self.num_flowers + 1
#         probs = [[0.0]*n for _ in range(n)]

#         # Build set of TSP edges (i->j)
#         tsp_edges = set()
#         for i in range(len(tsp_path) - 1):
#             tsp_edges.add((tsp_path[i], tsp_path[i+1]))

#         # Assign 0.6 if edge in TSP, else 0.1
#         for i in range(n):
#             for j in range(n):
#                 if i == j:
#                     probs[i][j] = 0.0  # no self-loop
#                 else:
#                     if (i, j) in tsp_edges:
#                         probs[i][j] = 0.6
#                     else:
#                         probs[i][j] = 0.1
#             # Normalize row i
#             row_sum = sum(probs[i])
#             if row_sum > 0:
#                 probs[i] = [val / row_sum for val in probs[i]]

#         self.transition_probabilities = probs

#     def _normalize_transition_probabilities(self):
#         """
#         Normalize each row to sum to 1.0
#         """
#         n = self.num_flowers + 1
#         for i in range(n):
#             row_sum = sum(self.transition_probabilities[i])
#             if row_sum > 0:
#                 self.transition_probabilities[i] = [
#                     p / row_sum for p in self.transition_probabilities[i]
#                 ]

#     def _reinforce_tsp_edges(self, path):
#         """
#         Compare the distance of this path to best_distance.
#         If improved, multiply TSP-like edges by reinforcement_factor in all rows.
#         Then re-normalize.
#         """
#         path_dist = self._compute_path_distance(path)
#         if path_dist < self.best_distance:
#             # Found a new best path => apply reinforcement
#             self.best_distance = path_dist
#             n = self.num_flowers + 1
#             # Multiply edges > 0.5 by reinforcement
#             for i in range(n):
#                 for j in range(n):
#                     if self.transition_probabilities[i][j] > 0.5:
#                         self.transition_probabilities[i][j] *= self.reinforcement_factor
            
#             # Re-normalize after multiplying
#             self._normalize_transition_probabilities()

#     def _choose_next_flower(self, current_index):
#         """
#         Randomly choose the next index (1..N) or possibly 0 if we allowed immediate nest returns.
#         However, to see real 'wandering', let's exclude returning to nest mid-trip
#         by ignoring the transition probability to j=0 if we want more realistic 'flight'.
#         """
#         row_probs = self.transition_probabilities[current_index]

#         # Option A: Let the nest be possible at any time => repeated "home" visits
#         # Option B: Exclude nest from mid-trip so the bee doesn't come home early
#         # We'll do Option B: if current_index != 0, then nest prob = 0
#         masked_probs = row_probs[:]
#         if current_index != 0:
#             masked_probs[0] = 0  # disallow returning to nest mid-way

#         # Re-normalize masked
#         s = sum(masked_probs)
#         if s <= 1e-9:
#             # fallback: uniform distribution among 1..N
#             n = self.num_flowers
#             uniform = 1.0 / n
#             masked_probs = [0.0]*(n+1)
#             for j in range(1, n+1):
#                 masked_probs[j] = uniform
#         else:
#             masked_probs = [p/s for p in masked_probs]

#         # Sample next index
#         next_index = np.random.choice(len(masked_probs), p=masked_probs)
#         return next_index

#     def _compute_path_distance(self, path):
#         return sum(
#             self.distances[path[i]][path[i+1]] for i in range(len(path)-1)
#         )

#     def simulate(self):
#         """
#         1) Solve TSP for reference path & distance
#         2) Initialize transition_probs accordingly
#         3) For each round trip:
#            - Let the bee do 'steps_per_trip' moves starting at nest=0
#            - Then it returns to nest
#            - Compare total distance to best_distance => reinforce TSP edges if improved
#         """
#         all_positions = [self.start_position] + self.flower_positions

#         # 1) TSP
#         tsp_path, tsp_distance = self._solve_tsp()
#         print(f"[TSP] Optimal path: {[all_positions[i] for i in tsp_path]}, Distance={tsp_distance:.2f}")

#         # 2) Initialize transition probs
#         self._init_transition_probabilities(tsp_path)

#         # For plotting
#         all_paths = []
#         all_distances = []

#         for r in range(1, self.max_round_trips+1):
#             # Round trip: start at nest
#             path = [0]
#             for _ in range(self.steps_per_trip):
#                 current_index = path[-1]
#                 next_index = self._choose_next_flower(current_index)
#                 path.append(next_index)

#             # Return to nest explicitly
#             path.append(0)

#             # Distance
#             dist = self._compute_path_distance(path)
#             all_paths.append(path)
#             all_distances.append(dist)

#             # Log data
#             self.round_trip_data.append({
#                 "Round Trip": r,
#                 "Path Indices": path,
#                 "Path Coordinates": [all_positions[i] for i in path],
#                 "Distance": dist
#             })

#             # 3) Reinforce
#             self._reinforce_tsp_edges(path)

#         self._plot_paths(all_positions, all_paths, all_distances)
#         return self.round_trip_data

#     def _plot_paths(self, all_positions, paths, distances):
#         """
#         Visualize:
#           - Each round trip path as a separate figure
#           - A final figure for distance vs. round trip
#         """
#         for i, path in enumerate(paths, start=1):
#             plt.figure(figsize=(6, 6))
#             x = [all_positions[idx][0] for idx in path]
#             y = [all_positions[idx][1] for idx in path]
#             plt.plot(x, y, marker="o", label=f"Round Trip {i}")
#             # Mark nest in blue, flowers in red
#             plt.scatter(all_positions[0][0], all_positions[0][1], c="blue", s=80, label="Nest")
#             plt.scatter(
#                 [p[0] for p in all_positions[1:]],
#                 [p[1] for p in all_positions[1:]],
#                 c="red", s=80, label="Flowers"
#             )
#             plt.title(f"Round Trip {i} (Distance={distances[i-1]:.2f})")
#             plt.xlabel("X")
#             plt.ylabel("Y")
#             plt.grid()
#             plt.legend()
#             plt.show()

#         # Plot total distance per round trip
#         plt.figure(figsize=(8, 5))
#         plt.plot(range(1, len(distances)+1), distances, marker="o")
#         plt.title("Total Distance Over Round Trips")
#         plt.xlabel("Round Trip")
#         plt.ylabel("Distance")
#         plt.grid()
#         plt.show()

#     def export_to_excel(self, filename="bee_simulation_results_with_revisits.xlsx"):
#         data_rows = []
#         for record in self.round_trip_data:
#             path_str = " -> ".join(map(str, record["Path Indices"]))
#             data_rows.append({
#                 "Round Trip": record["Round Trip"],
#                 "Path Indices": path_str,
#                 "Path Coordinates": record["Path Coordinates"],
#                 "Distance": record["Distance"]
#             })
#         df = pd.DataFrame(data_rows)
#         df.to_excel(filename, index=False)
#         print(f"[Export] Results saved to {filename}")


# # ----------------
# # USAGE EXAMPLE
# # ----------------
# if __name__ == "__main__":
#     start_position = (0, -12)
#     flower_positions = [
#         (-4, -4),  # Flower 1
#         (-4, 12),  # Flower 2
#         (0, 20),   # Flower 3
#         (4, 12),   # Flower 4
#         (4, -4),   # Flower 5
#     ]

#     # Create BeeModel with, say, 8 steps per trip => we allow revisits
#     bee_model = BeeModel(
#         start_position=start_position,
#         flower_positions=flower_positions,
#         reinforcement_factor=2.0,
#         max_round_trips=30,
#         steps_per_trip=8
#     )
    
#     results = bee_model.simulate()
#     bee_model.export_to_excel("bee_simulation_results_with_revisits.xlsx")
#     print("Simulation complete. Bee can revisit flowers; watch how the path evolves!")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations

class BeeModel:
    def __init__(
        self,
        start_position,
        flower_positions,
        reinforcement_factor=2.0,
        max_round_trips=1000,
        steps_per_trip=None
    ):
        """
        :param start_position: The coordinates of the nest, e.g. (x, y).
        :param flower_positions: A list of (x, y) positions for flowers.
        :param reinforcement_factor: Factor by which we reinforce TSP transitions
                                     when we find a better path.
        :param max_round_trips: Number of round trips the bee will take.
        :param steps_per_trip: How many 'moves' the bee makes before returning to the nest.
                               If None, defaults to number_of_flowers.
        """
        self.start_position = start_position
        self.flower_positions = flower_positions
        self.num_flowers = len(flower_positions)
        self.reinforcement_factor = reinforcement_factor
        self.max_round_trips = max_round_trips
        
        # If user doesn't specify steps per trip, default to #flowers
        self.steps_per_trip = steps_per_trip or self.num_flowers
        
        # 0 = nest, 1..N = flowers
        self.distances = self._calculate_distance_matrix()  
        
        # Will hold transition probabilities: (N+1) x (N+1)
        # Initialized after TSP is solved
        self.transition_probabilities = None
        
        # Data logging
        self.round_trip_data = []
        self.best_distance = float("inf")  # Track shortest total distance so far

    def _calculate_distance_matrix(self):
        """Compute a 2D matrix of pairwise distances among nest + all flowers."""
        all_positions = [self.start_position] + self.flower_positions
        n = len(all_positions)
        dist_matrix = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = np.linalg.norm(
                    np.array(all_positions[i]) - np.array(all_positions[j])
                )
        return dist_matrix

    def _solve_tsp(self):
        """
        Brute-force TSP solver (works for small #flowers).
        Returns:
          tsp_path: list of indices (0 = nest, 1..N = flowers) including 0 at start & end.
          tsp_distance: total distance of that route.
        """
        points = list(range(1, self.num_flowers + 1))
        best_distance = float("inf")
        best_path = []
        for perm in permutations(points):
            path = [0] + list(perm) + [0]
            dist = sum(self.distances[path[i]][path[i+1]] for i in range(len(path) - 1))
            if dist < best_distance:
                best_distance = dist
                best_path = path
        return best_path, best_distance

    def _init_transition_probabilities(self, tsp_path):
        """
        Initialize transition probabilities matrix (N+1 x N+1).
        If (i->j) is on the TSP path, give base prob = 0.6.
        Otherwise, base prob = 0.1.
        Then row-normalize.
        """
        n = self.num_flowers + 1
        probs = [[0.0]*n for _ in range(n)]

        # Build set of TSP edges (i->j)
        tsp_edges = set()
        for i in range(len(tsp_path) - 1):
            tsp_edges.add((tsp_path[i], tsp_path[i+1]))

        # Assign 0.6 if edge in TSP, else 0.1
        for i in range(n):
            for j in range(n):
                if i == j:
                    probs[i][j] = 0.0  # no self-loop
                else:
                    if (i, j) in tsp_edges:
                        probs[i][j] = 0.6
                    else:
                        probs[i][j] = 0.1
            # Normalize row i
            row_sum = sum(probs[i])
            if row_sum > 0:
                probs[i] = [val / row_sum for val in probs[i]]

        self.transition_probabilities = probs

    def _normalize_transition_probabilities(self):
        """
        Normalize each row to sum to 1.0
        """
        n = self.num_flowers + 1
        for i in range(n):
            row_sum = sum(self.transition_probabilities[i])
            if row_sum > 0:
                self.transition_probabilities[i] = [
                    p / row_sum for p in self.transition_probabilities[i]
                ]

    def _reinforce_tsp_edges(self, path):
        """
        Compare the distance of this path to best_distance.
        If improved, multiply TSP-like edges by reinforcement_factor in all rows.
        Then re-normalize.
        """
        path_dist = self._compute_path_distance(path)
        if path_dist < self.best_distance:
            # Found a new best path => apply reinforcement
            self.best_distance = path_dist
            n = self.num_flowers + 1
            # Multiply edges > 0.5 by reinforcement
            for i in range(n):
                for j in range(n):
                    if self.transition_probabilities[i][j] > 0.5:
                        self.transition_probabilities[i][j] *= self.reinforcement_factor
            
            # Re-normalize after multiplying
            self._normalize_transition_probabilities()

    # def _choose_next_flower(self, current_index, visited):
    #     """
    #     Randomly choose the next flower or return to the nest.
    #     Ensure all flowers are eventually visited.
    #     """
    #     # If all flowers are visited, allow any flower or the nest
    #     if len(visited) == self.num_flowers:
    #         available_choices = list(range(1, self.num_flowers + 1)) + [0]
    #     else:
    #         # Prioritize unvisited flowers
    #         unvisited = [f for f in range(1, self.num_flowers + 1) if f not in visited]
    #         available_choices = unvisited

    #     # Use the row of transition probabilities for weighted choice
    #     row_probs = self.transition_probabilities[current_index]
    #     masked_probs = [row_probs[j] if j in available_choices else 0 for j in range(len(row_probs))]

    #     # Normalize probabilities
    #     total_prob = sum(masked_probs)
    #     if total_prob > 0:
    #         masked_probs = [p / total_prob for p in masked_probs]
    #     else:
    #         # Fallback to uniform selection among available choices
    #         uniform_prob = 1.0 / len(available_choices)
    #         masked_probs = [uniform_prob if j in available_choices else 0 for j in range(len(row_probs))]

    #     # Choose the next index
    #     next_index = np.random.choice(len(masked_probs), p=masked_probs)
    #     return next_index

    def _choose_next_flower(self, current_index, visited):
        """
        Randomly choose the next flower or return to the nest.
        Revisits are allowed during the trip, with higher probability for unvisited flowers.
        """
        unvisited = [f for f in range(1, self.num_flowers + 1) if f not in visited]

        # Allow revisits during the trip but give higher weight to unvisited flowers
        all_choices = list(range(1, self.num_flowers + 1))  # Include all flowers
        if len(visited) == self.num_flowers:
            # All flowers visited, include the nest as an option
            all_choices.append(0)

        # Adjust probabilities to favor unvisited flowers
        row_probs = self.transition_probabilities[current_index]
        masked_probs = [
            row_probs[j] * (1.5 if j in unvisited else 0.5) for j in all_choices
        ]

        # Normalize probabilities
        total_prob = sum(masked_probs)
        if total_prob > 0:
            masked_probs = [p / total_prob for p in masked_probs]
        else:
            # Fallback to uniform distribution if probabilities sum to zero
            uniform_prob = 1.0 / len(all_choices)
            masked_probs = [uniform_prob for _ in all_choices]

        # Choose the next index
        next_index = np.random.choice(all_choices, p=masked_probs)
        return next_index

    def _compute_path_distance(self, path):
        return sum(
            self.distances[path[i]][path[i+1]] for i in range(len(path)-1)
        )

    # def simulate(self):
    #     """
    #     1) Solve TSP for reference path & distance
    #     2) Initialize transition_probs accordingly
    #     3) For each round trip:
    #     - Allow revisits to flowers but ensure all flowers are visited at least once
    #     - Compare total distance to best_distance => reinforce TSP edges if improved
    #     """
    #     all_positions = [self.start_position] + self.flower_positions

    #     # 1) Solve TSP
    #     tsp_path, tsp_distance = self._solve_tsp()
    #     print(f"[TSP] Optimal path: {[all_positions[i] for i in tsp_path]}, Distance={tsp_distance:.2f}")

    #     # 2) Initialize transition probabilities
    #     self._init_transition_probabilities(tsp_path)

    #     # For plotting
    #     all_paths = []
    #     all_distances = []

    #     for r in range(1, self.max_round_trips + 1):
    #         # Round trip: start at nest
    #         path = [0]
    #         visited = set()

    #         for _ in range(self.steps_per_trip):
    #             current_index = path[-1]
    #             next_index = self._choose_next_flower(current_index, visited)
    #             path.append(next_index)

    #             # Track visited flowers (ignore nest)
    #             if next_index != 0:
    #                 visited.add(next_index)

    #         # Ensure all flowers are visited at least once
    #         while len(visited) < self.num_flowers:
    #             current_index = path[-1]
    #             next_index = self._choose_next_flower(current_index, visited)
    #             path.append(next_index)
    #             if next_index != 0:
    #                 visited.add(next_index)

    #         # Return to nest explicitly
    #         path.append(0)

    #         # Calculate distance
    #         dist = self._compute_path_distance(path)
    #         all_paths.append(path)
    #         all_distances.append(dist)

    #         # Log data
    #         self.round_trip_data.append({
    #             "Round Trip": r,
    #             "Path Indices": path,
    #             "Path Coordinates": [all_positions[i] for i in path],
    #             "Distance": dist
    #         })

    #         # Reinforce TSP edges
    #         self._reinforce_tsp_edges(path)

    #     # Plot paths and distances
    #     self._plot_paths(all_positions, all_paths, all_distances)
    #     return self.round_trip_data

    def simulate(self):
        """
        1) Solve TSP for reference path & distance
        2) Initialize transition_probs accordingly
        3) For each round trip:
        - Allow revisits to flowers but ensure all flowers are visited at least once
        - Compare total distance to best_distance => reinforce TSP edges if improved
        """
        all_positions = [self.start_position] + self.flower_positions

        # 1) Solve TSP
        tsp_path, tsp_distance = self._solve_tsp()
        print(f"[TSP] Optimal path: {[all_positions[i] for i in tsp_path]}, Distance={tsp_distance:.2f}")

        # 2) Initialize transition probabilities
        self._init_transition_probabilities(tsp_path)

        # For plotting
        all_paths = []
        all_distances = []

        for r in range(1, self.max_round_trips + 1):
            # Round trip: start at nest
            path = [0]
            visited = set()

            # Allow revisits but ensure all flowers are visited
            for _ in range(self.steps_per_trip):
                current_index = path[-1]
                next_index = self._choose_next_flower(current_index, visited)
                path.append(next_index)

                # Mark flowers as visited (ignore nest)
                if next_index != 0:
                    visited.add(next_index)

            # Ensure all flowers are visited before ending the trip
            while len(visited) < self.num_flowers:
                current_index = path[-1]
                next_index = self._choose_next_flower(current_index, visited)
                path.append(next_index)
                if next_index != 0:
                    visited.add(next_index)

            # Return to nest explicitly
            path.append(0)

            # Calculate distance
            dist = self._compute_path_distance(path)
            all_paths.append(path)
            all_distances.append(dist)

            # Log data
            self.round_trip_data.append({
                "Round Trip": r,
                "Path Indices": path,
                "Path Coordinates": [all_positions[i] for i in path],
                "Distance": dist
            })

            # Reinforce TSP edges
            self._reinforce_tsp_edges(path)

        # Plot paths and distances
        self._plot_paths(all_positions, all_paths, all_distances)
        return self.round_trip_data

    def _plot_paths(self, all_positions, paths, distances):
        """
        Visualize:
          - Each round trip path as a separate figure
          - A final figure for distance vs. round trip
        """
        # for i, path in enumerate(paths, start=1):
        #     plt.figure(figsize=(6, 6))
        #     x = [all_positions[idx][0] for idx in path]
        #     y = [all_positions[idx][1] for idx in path]
        #     plt.plot(x, y, marker="o", label=f"Round Trip {i}")
        #     #Mark nest in blue, flowers in red
        #     plt.scatter(all_positions[0][0], all_positions[0][1], c="blue", s=80, label="Nest")
        #     plt.scatter(
        #         [p[0] for p in all_positions[1:]],
        #         [p[1] for p in all_positions[1:]],
        #         c="red", s=80, label="Flowers"
        #     )
        #     plt.title(f"Round Trip {i} (Distance={distances[i-1]:.2f})")
        #     plt.xlabel("X")
        #     plt.ylabel("Y")
        #     plt.grid()
        #     plt.legend()
        #     plt.show()

        # Plot total distance per round trip
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(distances)+1), distances, marker="o")
        plt.title("Total Distance Over Round Trips")
        plt.xlabel("Round Trip")
        plt.ylabel("Distance")
        plt.grid()
        plt.show()

    def export_to_excel(self, filename="bee_simulation_results_with_revisits.xlsx"):
        data_rows = []
        for record in self.round_trip_data:
            path_str = " -> ".join(map(str, record["Path Indices"]))
            data_rows.append({
                "Round Trip": record["Round Trip"],
                "Path Indices": path_str,
                "Path Coordinates": record["Path Coordinates"],
                "Distance": record["Distance"]
            })
        df = pd.DataFrame(data_rows)
        df.to_excel(filename, index=False)
        print(f"[Export] Results saved to {filename}")


# ----------------
# USAGE EXAMPLE
# ----------------
if __name__ == "__main__":
    start_position = (0, -12)
    flower_positions = [
        (-4, -4),  # Flower 1
        (-4, 12),  # Flower 2
        #(5, 20),   # Flower 3
        (4, 12),   # Flower 4
        (6,5),
        (4, -4),   # Flower 5
        # (20,0)
    ]

    # Create BeeModel
    bee_model = BeeModel(
        start_position=start_position,
        flower_positions=flower_positions,
        reinforcement_factor=2.0,
        max_round_trips=1000,
    )
    
    results = bee_model.simulate()
    bee_model.export_to_excel("bee_simulation_results_with_revisits.xlsx")
    print("Simulation complete. Results exported.")