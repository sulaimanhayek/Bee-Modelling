# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from itertools import permutations
# from BeeModel import BeeModel


# class MultiBeeModel:
#     def __init__(
#         self,
#         start_position,
#         flower_positions,
#         num_bees=7,
#         reinforcement_factor=2.0,
#         max_round_trips=30,
#         steps_per_trip=None,
#     ):
#         """
#         :param start_position: The coordinates of the nest, e.g., (x, y).
#         :param flower_positions: A list of (x, y) positions for flowers.
#         :param num_bees: Number of bees in the simulation.
#         :param reinforcement_factor: Factor for reinforcing TSP-based transitions.
#         :param max_round_trips: Maximum number of round trips for each bee.
#         :param steps_per_trip: Maximum moves allowed per trip for each bee.
#         """
#         self.start_position = start_position
#         self.flower_positions = flower_positions
#         self.num_bees = num_bees
#         self.reinforcement_factor = reinforcement_factor
#         self.max_round_trips = max_round_trips
#         self.steps_per_trip = steps_per_trip or len(flower_positions)
#         self.bees = [
#             BeeModel(
#                 start_position=start_position,
#                 flower_positions=flower_positions,
#                 reinforcement_factor=reinforcement_factor,
#                 max_round_trips=max_round_trips,
#                 steps_per_trip=self.steps_per_trip,
#             )
#             for _ in range(num_bees)
#         ]
#         self.aggregate_results = []

#     def simulate(self):
#         """
#         Run the simulation for all bees.
#         """
#         for bee_index, bee in enumerate(self.bees, start=1):
#             print(f"Simulating Bee {bee_index}...")
#             bee_results = bee.simulate()
#             self.aggregate_results.append(
#                 {"Bee": bee_index, "Results": bee_results}
#             )

#     def export_to_excel(self, filename="multi_bee_simulation_results.xlsx"):
#         """
#         Export the results of all bees to an Excel file.
#         """
#         data = []
#         for bee_data in self.aggregate_results:
#             bee_index = bee_data["Bee"]
#             for record in bee_data["Results"]:
#                 path_str = " -> ".join(map(str, record["Path Indices"]))
#                 data.append(
#                     {
#                         "Bee": bee_index,
#                         "Round Trip": record["Round Trip"],
#                         "Path Indices": path_str,
#                         "Path Coordinates": record["Path Coordinates"],
#                         "Distance": record["Distance"],
#                     }
#                 )
#         df = pd.DataFrame(data)
#         df.to_excel(filename, index=False)
#         print(f"[Export] Results saved to {filename}")

#     def visualize_aggregate_distances(self):
#         """
#         Plot the total distances of all bees across their round trips.
#         """
#         plt.figure(figsize=(10, 6))
#         for bee_data in self.aggregate_results:
#             bee_index = bee_data["Bee"]
#             distances = [
#                 record["Distance"] for record in bee_data["Results"]
#             ]
#             plt.plot(
#                 range(1, len(distances) + 1),
#                 distances,
#                 marker="o",
#                 label=f"Bee {bee_index}",
#             )
#         plt.title("Total Distance Over Round Trips (All Bees)")
#         plt.xlabel("Round Trip")
#         plt.ylabel("Total Distance")
#         plt.legend()
#         plt.grid()
#         plt.show()


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

#     # Initialize MultiBeeModel
#     multi_bee_model = MultiBeeModel(
#         start_position=start_position,
#         flower_positions=flower_positions,
#         num_bees=7,
#         reinforcement_factor=2.0,
#         max_round_trips=5,
#     )

#     # Simulate for all bees
#     multi_bee_model.simulate()

#     # Export results to Excel
#     multi_bee_model.export_to_excel("multi_bee_simulation_results.xlsx")

#     # Visualize aggregate distances
#     #multi_bee_model.visualize_aggregate_distances()

#     print("Simulation complete. Results exported.")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from BeeModel import BeeModel


class MultiBeeModel:
    def __init__(
        self,
        start_position,
        flower_positions,
        num_bees=7,
        reinforcement_factor=2.0,
        max_round_trips=30,
        steps_per_trip=None,
    ):
        self.start_position = start_position
        self.flower_positions = flower_positions
        self.num_bees = num_bees
        self.reinforcement_factor = reinforcement_factor
        self.max_round_trips = max_round_trips
        self.steps_per_trip = steps_per_trip or len(flower_positions)
        self.bees = [
            BeeModel(
                start_position=start_position,
                flower_positions=flower_positions,
                reinforcement_factor=reinforcement_factor,
                max_round_trips=max_round_trips,
                steps_per_trip=self.steps_per_trip,
            )
            for _ in range(num_bees)
        ]
        self.aggregate_results = []

    def simulate(self):
        """
        Run the simulation for all bees.
        """
        for bee_index, bee in enumerate(self.bees, start=1):
            print(f"Simulating Bee {bee_index}...")
            bee_results = bee.simulate()
            self.aggregate_results.append(
                {"Bee": bee_index, "Results": bee_results}
            )

    def export_to_excel(self, filename="multi_bee_simulation_results.xlsx"):
        """
        Export the results of all bees to an Excel file.
        """
        data = []
        for bee_data in self.aggregate_results:
            bee_index = bee_data["Bee"]
            for record in bee_data["Results"]:
                path_str = " -> ".join(map(str, record["Path Indices"]))
                data.append(
                    {
                        "Bee ID": bee_index,
                        "Round Trip": record["Round Trip"],
                        "Path Indices": path_str,
                        "Path Coordinates": record["Path Coordinates"],
                        "Distance": record["Distance"],
                    }
                )
        df = pd.DataFrame(data)
        df.to_excel(filename, index=False)
        print(f"[Export] Results saved to {filename}")

    def visualize_aggregate_distances(self):
        """
        Plot the total distances of all bees across their round trips.
        """
        plt.figure(figsize=(12, 8))
        for bee_data in self.aggregate_results:
            bee_index = bee_data["Bee"]
            distances = [
                record["Distance"] for record in bee_data["Results"]
            ]
            plt.plot(
                range(1, len(distances) + 1),
                distances,
                marker="o",
                label=f"Bee {bee_index}",
            )
        plt.title("Total Distance Over Round Trips (All Bees)")
        plt.xlabel("Round Trip")
        plt.ylabel("Total Distance")
        plt.legend()
        plt.grid()
        plt.show()


# ----------------
# USAGE EXAMPLE
# ----------------
if __name__ == "__main__":
    start_position = (0, -12)
    flower_positions = [
        (-4, -4),  # Flower 1
        (-4, 12),  # Flower 2
        (0, 20),   # Flower 3
        (4, 12),   # Flower 4
        (4, -4)   # Flower 5
    ]

    # Initialize MultiBeeModel
    multi_bee_model = MultiBeeModel(
        start_position=start_position,
        flower_positions=flower_positions,
        num_bees=7,
        reinforcement_factor=2.0,
        max_round_trips=30,
    )

    # Simulate for all bees
    multi_bee_model.simulate()

    # Export results to Excel
    multi_bee_model.export_to_excel("multi_bee_simulation_results.xlsx")

    # Visualize aggregate distances
    multi_bee_model.visualize_aggregate_distances()

    print("Simulation complete. Results exported and visualized.")


