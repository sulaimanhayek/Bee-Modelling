# import matplotlib.pyplot as plt

# def visualize_bee_path(path, nest_position, flower_positions):
#     """
#     Plots the bee's path on a 2D axis, including nest and flowers.
#     """
#     # Extract x, y for the path
#     x_coords = [p[0] for p in path]
#     y_coords = [p[1] for p in path]

#     # Separate nest coords
#     nest_x, nest_y = nest_position

#     # Separate flower coords
#     fx = [f[0] for f in flower_positions]
#     fy = [f[1] for f in flower_positions]

#     plt.figure(figsize=(6, 6))

#     # Plot the bee path
#     plt.plot(x_coords, y_coords, marker='o', color='blue', label='Bee Path')

#     # Plot the nest (as a red square)
#     plt.scatter(nest_x, nest_y, marker='s', s=100, color='red', label='Nest')

#     # Plot the flowers (as green stars)
#     plt.scatter(fx, fy, marker='*', s=150, color='green', label='Flowers')

#     # Add labels to flowers
#     for i, (fxi, fyi) in enumerate(flower_positions):
#         plt.text(fxi, fyi, f"Flower {i+1}", fontsize=9, ha='center', va='center')

#     # Add label to the nest
#     plt.text(nest_x, nest_y, "Nest", fontsize=9, ha='center', va='center', color='red')

#     # Styling
#     plt.title("Bee Foraging Path")
#     plt.xlabel("X-coordinate")
#     plt.ylabel("Y-coordinate")
#     plt.grid(True)
#     plt.legend()
#     plt.show()



import matplotlib.pyplot as plt

def visualize_bee_path(paths, nest_position, flower_positions, labels=None, title="Bee Path"):
    """
    Visualize the bee's path on a 2D plane.
    
    :param paths: List of paths to visualize (list of lists of tuples).
    :param nest_position: Tuple representing the nest position (x, y).
    :param flower_positions: List of tuples representing flower positions [(x1, y1), ...].
    :param labels: List of labels for each path.
    :param title: Title of the plot.
    """
    plt.figure(figsize=(8, 8))

    # Plot the nest
    plt.scatter(*nest_position, color='red', s=100, label='Nest')

    # Plot the flowers
    for flower in flower_positions:
        plt.scatter(*flower, color='blue', s=100, label='Flower')

    # Plot paths
    for i, path in enumerate(paths):
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        label = labels[i] if labels else f"Path {i+1}"
        plt.plot(path_x, path_y, marker='o', label=label)

    # Add labels and title
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()