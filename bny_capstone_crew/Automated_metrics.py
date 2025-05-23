import json
from collections import Counter
import numpy as np
import glob

meeting_date = "September 2024"
correct_vote = ["-0.50%"]

# Step 1: Get a list of all JSON files
json_files = glob.glob("rate_summary*.json")  # Adjust pattern to match your file names

# Step 2: Read all JSON files into a list
rate_summaries = []

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        rate_summaries.append(data)  # Append each JSON object to the list


# Dictionary to store votes by member
votes_by_member = {}
predictions_by_member = {}
all_votes = []
all_dates = []
all_metrics = []

# Loop through each JSON object
for summary in rate_summaries:
    for vote in summary["rate_votes"]:
        member = vote["member"]
        vote_value = vote["vote"]

        # Initialize list if member not in dictionary
        if member not in votes_by_member:
            votes_by_member[member] = []

        # Append vote to the corresponding member
        votes_by_member[member].append(vote_value)
        all_votes.append(vote_value)
    for prediction in summary["rate_predictions"]:
        member = prediction["member"]
        prediction_value = prediction["prediction"]

        # Initialize list if member not in dictionary
        if member not in predictions_by_member:
            predictions_by_member[member] = []

        # Append vote to the corresponding member
        predictions_by_member[member].append(prediction_value)
    all_dates.append(summary["exact_historical_dates_referenced"])
    all_metrics.append(summary["exact_metrics_mentioned"])

vote_agreement_rates = []
prediction_agreement_rates = []

for member in votes_by_member:
    vote_counts = Counter(votes_by_member[member])
    # Get the most common value and its count
    most_common_value, count = vote_counts.most_common(1)[0]
    vote_agreement_rates.append(100 * count / len(votes_by_member[member]))
    # Do the same for predictions
    prediction_counts = Counter(predictions_by_member[member])
    # Get the most common value and its count
    most_common_value, count = prediction_counts.most_common(1)[0]
    prediction_agreement_rates.append(100 * count / len(predictions_by_member[member]))


# print(vote_agreement_rates)
avg_vote_agreement = np.mean(vote_agreement_rates)
print(f"Meeting Date: {meeting_date}")
print(f"average agreement rate for votes: {avg_vote_agreement}%")
# print(prediction_agreement_rates)
avg_prediction_agreement = np.mean(prediction_agreement_rates)
print(
    f"average agreement rate for predictions: {np.round(avg_prediction_agreement,2)}%"
)
print(
    f"total agreement rate: {np.round(np.mean([avg_vote_agreement,avg_prediction_agreement]),2)}%"
)
correct_vote_pct = (
    100 * sum([vote in correct_vote for vote in all_votes]) / len(all_votes)
)
print(f"correct rate vote percentage = {correct_vote_pct}%")

print()
print("All dates mentioned:")
for dates in all_dates:
    print(dates)

print()
print("All metrics mentioned:")
for metrics in all_metrics:
    print(metrics)
