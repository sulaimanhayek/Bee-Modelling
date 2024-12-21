import pandas as pd

# Export bee simulation results to Excel
def export_results_to_excel(bee_paths, total_distances, file_name="bee_simulation_results.xlsx"):
    data = []
    for bee_id, (path, distance) in enumerate(zip(bee_paths, total_distances), start=1):
        visit_counts = {flower_id: path.count(flower_id) for flower_id in set(path)}
        for step, flower_id in enumerate(path):
            data.append({
                "Bee ID": bee_id,
                "Step": step + 1,
                "Flower ID": flower_id,
                "Visits (up to this step)": visit_counts[flower_id],
                "Path from Home (Total Distance)": f"{distance:.2f}" if step == len(path) - 1 else "",
            })

    # Create a DataFrame
    df = pd.DataFrame(data) 

    # Export to Excel
    df.to_excel(file_name, index=False)
    print(f"Results exported to {file_name}")    