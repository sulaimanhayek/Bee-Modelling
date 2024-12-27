import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

class BeeModelHeuristic:
    def __init__(
        self,
        start_position,
        flower_positions,
        reinforcement_factor=2.0,
        max_round_trips=25,
        steps_per_trip=None,
        max_2opt_iterations=4
    ):
        """
        :param start_position: (x, y) for the nest.
        :param flower_positions: A list of (x, y) positions for flowers.
        :param reinforcement_factor: Factor by which we reinforce edges if a new best path is found.
        :param max_round_trips: Number of round trips in the simulation.
        :param steps_per_trip: Moves made before returning to the nest. If None, defaults to num_flowers.
        :param max_2opt_iterations: Maximum 2-opt improvement iterations for the initial route.
        """
        self.start_position = start_position
        self.flower_positions = flower_positions
        self.num_flowers = len(flower_positions)
        self.reinforcement_factor = reinforcement_factor
        self.max_round_trips = max_round_trips
        self.max_2opt_iterations = max_2opt_iterations
        self.steps_per_trip = steps_per_trip or self.num_flowers

        # Build distance matrix
        self.distances = self._calculate_distance_matrix()

        # Transition probabilities (will be initialized after TSP)
        self.transition_probabilities = None

        # Tracking
        self.round_trip_data = []
        self.best_distance = float("inf")

    def _calculate_distance_matrix(self):
        """Compute a 2D matrix of pairwise distances among (nest + all flowers)."""
        all_positions = [self.start_position] + self.flower_positions
        n = len(all_positions)
        dist_matrix = [[0.0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = np.linalg.norm(
                    np.array(all_positions[i]) - np.array(all_positions[j])
                )
        return dist_matrix

    # -------------------------------------------------------------------------
    # 1) Simple TSP Heuristic (2-opt improvement)
    # -------------------------------------------------------------------------
    def _iterative_tsp(self):
        """
        Generates an initial path (random) and uses 2-opt to improve it up to max_2opt_iterations.
        Returns (route, distance).
        """
        route = self._build_random_initial_route()
        best_distance = self._compute_path_distance(route)

        improved = True
        iteration = 0
        while improved and iteration < self.max_2opt_iterations:
            improved = False
            iteration += 1
            # We'll try 2-opt swaps, skipping the first and last index (nest)
            for i in range(1, len(route) - 2):
                for k in range(i+1, len(route) - 1):
                    new_route = self._two_opt_swap(route, i, k)
                    new_dist = self._compute_path_distance(new_route)
                    if new_dist < best_distance:
                        route = new_route
                        best_distance = new_dist
                        improved = True
                        break
                if improved:
                    break

        return route, best_distance

    def _build_random_initial_route(self):
        """Construct a random route [0, flower_1, ..., flower_n, 0]. (0 = nest)."""
        flowers = list(range(1, self.num_flowers + 1))
        random.shuffle(flowers)
        return [0] + flowers + [0]

    def _two_opt_swap(self, route, i, k):
        """
        2-opt swap: reverse the segment [i..k].
        Example: route = A B C D E, swap i=2, k=3 => A B (D C) E.
        """
        return route[:i] + route[i:k+1][::-1] + route[k+1:]

    def _compute_path_distance(self, path):
        """Compute total distance of given path via the distance matrix."""
        return sum(
            self.distances[path[i]][path[i+1]] for i in range(len(path) - 1)
        )

    # -------------------------------------------------------------------------
    # 2) Initialize Transition Probabilities from TSP
    # -------------------------------------------------------------------------
    def _init_transition_probabilities(self, tsp_path):
        """
        Create matrix of size (N+1)x(N+1), where N = number_of_flowers.
        If (i->j) in TSP path => base probability = 0.6
        Otherwise => base probability = 0.1
        Then row-normalize.
        """
        n = self.num_flowers + 1
        probs = [[0.0]*n for _ in range(n)]

        # Collect edges from TSP path
        tsp_edges = set()
        for i in range(len(tsp_path) - 1):
            tsp_edges.add((tsp_path[i], tsp_path[i+1]))

        for i in range(n):
            for j in range(n):
                if i == j:
                    probs[i][j] = 0.0
                elif (i, j) in tsp_edges:
                    probs[i][j] = 0.6
                else:
                    probs[i][j] = 0.1
            # Normalize row i
            row_sum = sum(probs[i])
            if row_sum > 0:
                probs[i] = [val / row_sum for val in probs[i]]

        self.transition_probabilities = probs

    # -------------------------------------------------------------------------
    # 3) Reinforce if New Best Path
    # -------------------------------------------------------------------------
    def _reinforce_tsp_edges(self, path):
        """
        If the path is shorter than our best_distance, multiply edges along path by reinforcement_factor (2.0).
        Then re-normalize each row.
        """
        path_dist = self._compute_path_distance(path)
        if path_dist < self.best_distance:
            self.best_distance = path_dist
            n = self.num_flowers + 1
            # Multiply edges in the path
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                self.transition_probabilities[a][b] *= self.reinforcement_factor
            # Re-normalize
            for i in range(n):
                row_sum = sum(self.transition_probabilities[i])
                if row_sum > 0:
                    self.transition_probabilities[i] = [
                        p / row_sum for p in self.transition_probabilities[i]
                    ]

    # -------------------------------------------------------------------------
    # 4) Simulation Step: Choose Next Flower
    # -------------------------------------------------------------------------
    def _choose_next_flower(self, current_index, visited):
        """
        We pick the next flower (or nest if all visited) stochastically from self.transition_probabilities,
        but apply a bonus to unvisited flowers (+50%) and a penalty to visited flowers (-50%).
        """
        unvisited = [f for f in range(1, self.num_flowers + 1) if f not in visited]
        all_choices = list(range(1, self.num_flowers + 1))
        if len(visited) == self.num_flowers:
            all_choices.append(0)  # allow returning to nest

        base_probs = self.transition_probabilities[current_index]
        masked_probs = []
        for j in all_choices:
            if j in unvisited:
                # bonus
                masked_probs.append(base_probs[j] * 1.5)
            else:
                # penalty
                masked_probs.append(base_probs[j] * 0.9)

        total = sum(masked_probs)
        if total > 0:
            masked_probs = [p / total for p in masked_probs]
        else:
            masked_probs = [1.0 / len(all_choices)] * len(all_choices)

        return np.random.choice(all_choices, p=masked_probs)

    # -------------------------------------------------------------------------
    # 5) Main Simulation
    # -------------------------------------------------------------------------
    def simulate(self):
        """
        1) Build a heuristic TSP path (2-opt).
        2) Initialize transition probabilities.
        3) For each round-trip:
           - Start from nest (0)
           - Move steps_per_trip times (with revisits allowed, but unvisited favored)
           - Ensure all flowers visited at least once, then return to nest
           - Log path/distance
           - If best => reinforce
        4) Plot each path
        5) Plot distance vs. round trip
        """
        all_positions = [self.start_position] + self.flower_positions

        # (1) Heuristic TSP
        tsp_path, tsp_distance = self._iterative_tsp()
        print(f"[Heuristic TSP] Path={tsp_path}, Distance={tsp_distance:.2f}")

        # (2) Init transition probabilities
        self._init_transition_probabilities(tsp_path)

        # For final plot
        distances_each_trip = []

        for r in range(1, self.max_round_trips + 1):
            path = [0]  # nest
            visited = set()

            # Move step by step
            for _ in range(self.steps_per_trip):
                current_index = path[-1]
                next_index = self._choose_next_flower(current_index, visited)
                path.append(next_index)
                if next_index != 0:
                    visited.add(next_index)

            # Keep going if not visited all flowers
            while len(visited) < self.num_flowers:
                current_index = path[-1]
                next_index = self._choose_next_flower(current_index, visited)
                path.append(next_index)
                if next_index != 0:
                    visited.add(next_index)

            # Return to nest
            path.append(0)

            # Distance
            dist = self._compute_path_distance(path)
            distances_each_trip.append(dist)

            # Log
            self.round_trip_data.append({
                "Round Trip": r,
                "Path Indices": path,
                "Path Coordinates": [all_positions[i] for i in path],
                "Distance": dist
            })

            # Possibly reinforce
            self._reinforce_tsp_edges(path)

            # Plot path
            self._plot_single_path(all_positions, path, r)

        # Plot distances over round trips
        self._plot_distance_vs_round_trip(distances_each_trip)
        return self.round_trip_data

    def _plot_single_path(self, all_positions, path, round_id):
        """Plot a single round-trip path (creates a separate figure)."""
        coords = [all_positions[i] for i in path]
        x, y = zip(*coords)

        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker="o", linestyle="-", markersize=8)
        for idx, (xi, yi) in enumerate(coords):
            label = "Nest" if path[idx] == 0 else f"Flower {path[idx]}"
            plt.text(xi, yi, label, fontsize=9, ha='right')

        plt.title(f"Bee Path - Round {round_id}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.show()

    def _plot_distance_vs_round_trip(self, distances):
        """After all round trips, show how distance changes over time."""
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(distances)+1), distances, marker="o")
        plt.title("Distance Over Round Trips")
        plt.xlabel("Round Trip")
        plt.ylabel("Distance")
        plt.grid()
        plt.show()

    def export_to_excel(self, filename="bee_simulation_results_add_flower.xlsx"):
        """Export round_trip_data to Excel."""
        data_rows = []
        for record in self.round_trip_data:
            path_str = " -> ".join(str(x) for x in record["Path Indices"])
            data_rows.append({
                "Round Trip": record["Round Trip"],
                "Path Indices": path_str,
                "Distance": record["Distance"],
                "Path Coordinates": record["Path Coordinates"]
            })
        df = pd.DataFrame(data_rows)
        df.to_excel(filename, index=False)
        print(f"[Export] Results saved to {filename}")


def main():
    """Example usage of BeeModelHeuristic."""
    # Lihoreau et al. 2011, 2012b
    start_position = (0, -12)
    flower_positions = [
        (-4, -4),  # Flower 1
        (-5, 12),  # Flower 2
        #(0, 20),   # Flower 3
        (5, 12),   # Flower 4
        (8,5), # Additional, flower 6
        (4, -4)    # Flower 5
    ]
    # Lihoreau et al. 2010
    # start_position = (0, 0)
    # flower_positions = [
    #     (10, -2),  # Flower 1
    #     (9, 10),  # Flower 2
    #     (6, 15),   # Flower 3
    #     (-4, 15),   # Flower 4
    # ]

    bee_model = BeeModelHeuristic(
        start_position=start_position,
        flower_positions=flower_positions,
        reinforcement_factor=1.4,  # how much to boost edges of better path
        max_round_trips=25,
        steps_per_trip=None,       # let the code use num_flowers
        max_2opt_iterations=4      # how many times to refine the initial TSP route
    )

    # Run simulation
    results = bee_model.simulate()

    # Export to Excel
    bee_model.export_to_excel("bee_simulation_results_add_flower.xlsx")
    print("Simulation complete, data exported to 'bee_simulation_results.xlsx'.")


if __name__ == "__main__":
    main()