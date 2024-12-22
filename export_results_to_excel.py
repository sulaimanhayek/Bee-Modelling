import pandas as pd

def export_results_to_excel(bee_bouts, bee_distances_per_bout, output_filename="bee_simulation_results.xlsx"):
    """
    Exports bee simulation results to an Excel file.

    Args:
        bee_bouts (list): A list of bouts for each bee. Each bout is a list of visited sites.
        bee_distances_per_bout (list): A list of distances for each bout per bee.
        output_filename (str): The name of the Excel file to save the results.
    """
    all_data = []

    for bee_index, (bouts, distances) in enumerate(zip(bee_bouts, bee_distances_per_bout)):
        for bout_index, (path, bout_distance) in enumerate(zip(bouts, distances)):
            # Handle path structure: extract flower IDs if it's a list of [flower_id, time]
            if path and isinstance(path[0], list):
                # Extract flower IDs only
                flower_ids = [step[0] for step in path]
            else:
                # Path is already a list of flower IDs
                flower_ids = path
            
            # Calculate visit counts
            visit_counts = {flower_id: flower_ids.count(flower_id) for flower_id in set(flower_ids)}

            # Append row data
            all_data.append({
                "Bee": bee_index + 1,
                "Bout": bout_index + 1,
                "Path": str(path),
                "BoutDistance": bout_distance,
                "VisitCounts": str(visit_counts)
            })
    
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(all_data)

    # Export to Excel
    df.to_excel(output_filename, index=False)
    print(f"Results exported to {output_filename}")