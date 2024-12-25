# MultiBeeModelHeuristic.py

import matplotlib.pyplot as plt
from BeeModelHeuristic import BeeModelHeuristic
import numpy as np

def run_multi_bee_simulation(num_bees=7, max_round_trips=30):
    """
    Run a simulation for multiple bees (each is a BeeModelHeuristic instance).
    Plot distance vs. iteration for each bee and a cumulative line if desired.
    """
    # Define scenario (same for each bee here, but you could vary if desired)
    start_position = (0, -12)
    flower_positions = [
        (-4, -4),  # Flower 1
        (-4, 12),  # Flower 2
        (0, 20),   # Flower 3
        (4, 12),   # Flower 4
        #(6, 8),    # Additional
        (4, -4)    # Flower 5
    ]

    # Parameters for each BeeModelHeuristic
    reinforcement_factor = 1.4
    steps_per_trip = None
    max_2opt_iterations = 5

    # Store distance results for each bee (list of lists)
    # distances_per_bee[b] = [dist_round1, dist_round2, ..., dist_roundN]
    distances_per_bee = []

    # Run simulation for each bee
    for b in range(num_bees):
        print(f"--- Running simulation for Bee #{b+1} ---")
        bee = BeeModelHeuristic(
            start_position=start_position,
            flower_positions=flower_positions,
            reinforcement_factor=reinforcement_factor,
            max_round_trips=max_round_trips,
            steps_per_trip=steps_per_trip,
            max_2opt_iterations=max_2opt_iterations
        )
        # Run the simulation
        result_data = bee.simulate()
        # Extract distances from result_data
        # Each item in result_data is: {"Round Trip": r, "Path Indices": ..., "Distance": dist, ...}
        distances = [record["Distance"] for record in result_data]
        distances_per_bee.append(distances)

    # Now plot the distance vs. iteration for each bee
    plt.figure(figsize=(10, 6))
    rounds = range(1, max_round_trips + 1)

    for b in range(num_bees):
        plt.plot(rounds, distances_per_bee[b], marker="o", label=f"Bee {b+1}")

    # If you also want to plot a "cumulative" or "mean" line:
    # Cumulative: sum of distances across all bees at each round
    # Mean: average distance across all bees
    # Here we show an example of the "mean" line:
    distances_array = np.array(distances_per_bee)  # shape = (num_bees, max_round_trips)
    mean_distances = np.mean(distances_array, axis=0)
    plt.plot(rounds, mean_distances, marker="s", color="black", linestyle="--", 
             label="Mean of All Bees")

    plt.title("Distance vs. Round Trip for Multiple Bees")
    plt.xlabel("Round Trip")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Customize how many bees, how many max round trips, etc., as desired.
    run_multi_bee_simulation(num_bees=7, max_round_trips=25)


if __name__ == "__main__":
    main()


