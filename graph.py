import matplotlib.pyplot as plt
from datetime import datetime
import random

# Define project activities and their corresponding months
activities = {
    "Project Kickoff": {"start": "2023-08-01", "end": "2023-08-07"},
    "Data Collection": {"start": "2023-09-01", "end": "2023-09-15"},
    "Data Preprocessing": {"start": "2023-09-15", "end": "2023-09-30"},
    "Sentiment Analysis Model Selection": {"start": "2023-10-01", "end": "2023-10-15"},
    "Model Training and Evaluation": {"start": "2023-10-15", "end": "2023-10-31"},
    "Model Optimization": {"start": "2023-11-01", "end": "2023-11-15"},
    "Expand Sentiment Analysis Scope": {"start": "2023-11-15", "end": "2023-11-30"},
    "Visualization Development": {"start": "2023-12-01", "end": "2023-12-15"},
    "Data Visualization": {"start": "2023-12-15", "end": "2023-12-31"},
    "Roadmap and Progress Evaluation": {"start": "2024-01-01", "end": "2024-01-15"},
    "Stakeholder Communication": {"start": "2024-01-15", "end": "2024-01-31"},
    "Final Model Testing and Validation": {"start": "2024-02-01", "end": "2024-02-15"},
    "Documentation": {"start": "2024-02-15", "end": "2024-02-29"},
    "Project Completion": {"start": "2024-03-01", "end": "2024-03-15"},
}

# Filter activities to include only those within the specified timeline
start_date = datetime.strptime("2023-08-01", "%Y-%m-%d")
end_date = datetime.strptime("2024-03-31", "%Y-%m-%d")
activities = {activity: dates for activity, dates in activities.items() if start_date <= dates["start"] <= end_date}

# Generate random colors for each activity
activity_colors = {activity: f"#{random.randint(0, 0xFFFFFF):06x}" for activity in activities}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Convert date strings to datetime objects
for activity, dates in activities.items():
    dates["start"] = datetime.strptime(dates["start"], "%Y-%m-%d")
    dates["end"] = datetime.strptime(dates["end"], "%Y-%m-%d")

# Sort activities by start date
sorted_activities = dict(sorted(activities.items(), key=lambda x: x[1]["start"]))

# Plot stacked bars for each activity
bottom = None
for activity, dates in sorted_activities.items():
    ax.bar(
        activity,
        (dates["end"] - dates["start"]).days,
        bottom=bottom,
        color=activity_colors[activity],
        label=activity,
        edgecolor='black'
    )
    if bottom is None:
        bottom = [0] * len(sorted_activities)
    bottom = [b + (dates["end"] - dates["start"]).days for b in bottom]

# Format the date axis
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))

# Add grid lines and customize their appearance
ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', axis='y')

# Set labels and title
plt.xlabel('Timeline')
plt.ylabel('Cumulative Duration')
plt.title('Project Roadmap - Stacked Bar Chart')

# Beautify the layout
plt.tight_layout()

# Save the chart as an image (optional)
# plt.savefig('stacked_bar_chart.png')

# Display the chart
plt.show()
