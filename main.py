# # main.py

# from BeeModel import BeeModel
# from visualize import visualize_bee_path

# def main():
#     result = BeeModel(verbose=False)  # You can set verbose=True if you want logs

#     # Extract data we need
#     path = result["path"]
#     nest_position = result["nest_position"]
#     flower_positions = result["flower_positions"]

#     # Visualize
#     visualize_bee_path(path, nest_position, flower_positions)

# if __name__ == "__main__":
#     main()



# main.py

# from BeeModel import BeeModel
# from visualize import visualize_bee_path

# def main():
#     # Run the simulation
#     result = BeeModel(verbose=True)

#     # Extract data for visualization
#     scouting_path = result["scouting_path"]
#     decision_path = result["decision_path"]
#     nest_position = result["nest_position"]
#     flower_positions = result["flower_positions"]

#     # Visualize scouting path
#     print("\nVisualizing Scouting Path...")
#     visualize_bee_path(scouting_path, nest_position, flower_positions)

#     # Visualize decision-making path
#     print("\nVisualizing Decision Path...")
#     visualize_bee_path(decision_path, nest_position, flower_positions)

# if __name__ == "__main__":
#     main()



# with distance

# main.py

# def main():
#     result = BeeModel(verbose=True)
    
#     scouting_path = result.get("scouting_path")
#     if scouting_path is None:
#         print("No scouting path found in the result. Using empty list instead.")
#         scouting_path = []
    
#     decision_path = result.get("decision_path")
#     if decision_path is None:
#         print("No decision path found in the result. Using empty list instead.")
#         decision_path = []
    
#     nest_position = result.get("nest_position", (0, 0))  # default if not found
#     flower_positions = result.get("flower_positions", [])

#     # Now visualize
#     print("\nVisualizing Scouting Path...")
#     visualize_bee_path(scouting_path, nest_position, flower_positions)

#     print("\nVisualizing Decision Path...")
#     visualize_bee_path(decision_path, nest_position, flower_positions)


from visualize import visualize_bee_path
from BeeModel import BeeModel

def main():
    # Define the initial conditions
    nest_position = (0, -12)
    flower_positions = [
        (-4, -4), (-4, 12), (0, 20), (4, 12), (4, -4)
    ]

    # Initialize the BeeModel
    bee_model = BeeModel(current_position=nest_position, points=flower_positions)

    # Run the simulation
    is_initial_move = True
    while True:
        next_point = bee_model.move_to_next_point(is_initial_move=is_initial_move)
        is_initial_move = False  # Only the first move uses initial probabilities
        if next_point == nest_position and len(bee_model.visited) == len(flower_positions):
            break

    # Get results from BeeModel
    result = bee_model.get_results()

    # Visualize scouting path
    print("\nVisualizing Scouting Path...")
    visualize_bee_path(
        paths=[result["scouting_path"]],
        nest_position=result["nest_position"],
        flower_positions=result["flower_positions"],
        labels=["Scouting Path"],
        title="Bee Scouting Path"
    )

if __name__ == "__main__":
    main()