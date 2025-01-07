# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from datetime import datetime
import numpy as np

# Load the dataset (update the file path to match your environment)
file_path = './data/6_month_code_metrics.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Selecting features for clustering
features = [
    "star_count",
    "fork_count",
    "developer_count",
    "contributor_count",
    "active_developer_count_6_months",
    # Normalized 6-month metrics
    "commits_per_dev",
    "prs_per_dev",
    "issues_per_dev",
    "contributors_per_dev",
    "first_commit_date"
]


# Clean and preprocess date columns
data["first_commit_date"] = pd.to_datetime(data["first_commit_date"], errors="coerce")
data["last_commit_date"] = pd.to_datetime(data["last_commit_date"], errors="coerce")
data = data.dropna(subset=["first_commit_date", "last_commit_date"])
data["first_commit_date"] = data["first_commit_date"].apply(lambda x: x.toordinal())
data["last_commit_date"] = data["last_commit_date"].apply(lambda x: x.toordinal())

# Create normalized metrics
data['commits_per_dev'] = data['commit_count_6_months'] / data['active_developer_count_6_months']
data['prs_per_dev'] = data['merged_pull_request_count_6_months'] / data['active_developer_count_6_months']
data['issues_per_dev'] = data['closed_issue_count_6_months'] / data['active_developer_count_6_months']
data['contributors_per_dev'] = data['contributor_count_6_months'] / data['active_developer_count_6_months']

# Handle infinite values from division by zero
data = data.replace([np.inf, -np.inf], np.nan)

# Additional preprocessing steps
# 1. Log transform highly skewed features
skewed_features = ['star_count', 'fork_count', 'commits_per_dev', 'prs_per_dev', 'issues_per_dev']
for feature in skewed_features:
    data[feature] = np.log1p(data[feature])  # log1p handles zero values

# Drop rows with missing values in selected features
data_cleaned = data.dropna(subset=features)

# Ensure 'project_name' is included in the cleaned data
data_cleaned['project_name'] = data['project_name']

# Scaling the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(data_cleaned[features])

# Applying K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Adding cluster labels to the dataset
data_cleaned['cluster'] = clusters

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for PCA results and clusters, including project names
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df.index = data_cleaned.index  # Ensure indices match
pca_df['Cluster'] = clusters
pca_df['project_name'] = data_cleaned['project_name'].values  # Use .values to ensure direct assignment


# Ensure 'Cluster' is treated as a categorical variable
pca_df['Cluster'] = pca_df['Cluster'].astype(str)

# Define a custom pastel color sequence
custom_pastel_colors = ['#FFB3BA', '#FFDFBA', '#D7BDE2', '#BAFFC9', '#BAE1FF']

# Define darker equivalents of the pastel colors
darker_colors = ['#81C784', '#E57373', '#9B59B6', '#FFB74D', '#64B5F6']

# Define custom labels for each cluster
cluster_labels = {
    '0': 'Early-Stage Projects',
    '1': 'High-Engagement Projects',
    '2': 'Niche Growing Projects',
    '3': 'Actively Evolving Projects',
    '4': 'Mature Ecosystem Leaders',
}

# Apply the custom labels to the Cluster column
pca_df['Cluster'] = pca_df['Cluster'].map(cluster_labels)

# Interactive Plotly scatter plot with project names displayed above each dot
fig = px.scatter(
    pca_df,
    x='PCA1',
    y='PCA2',
    color='Cluster',
    title="Clustering of Projects (Plotly Interactive)",
    labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
    template="plotly",
    text='project_name',  # Display project_name as text on the plot
    color_discrete_sequence=darker_colors
)

# Update the layout to adjust text position
fig.update_traces(textposition='top center')

# Update layout to move the legend to the bottom
fig.update_layout(
    legend=dict(
        orientation="h",  # Horizontal orientation
        yanchor="bottom",
        y=-0.2,  # Adjust this value to move the legend further down
        xanchor="center",
        x=0.5
    ),
    width=800,
)

fig.update_traces(marker=dict(size=10))
fig.show()

pca_df.to_csv('./data/pca_results.csv', index=False)

# Convert ordinal dates back to datetime
try:
    data_cleaned['first_commit_date'] = data_cleaned["first_commit_date"].apply(lambda x: datetime.fromordinal(x))
    data_cleaned['last_commit_date'] = data_cleaned["last_commit_date"].apply(lambda x: datetime.fromordinal(x))
except Exception as e:
    print("Error converting ordinal dates back to datetime:", e)

# Summarize clusters
cluster_summary = data_cleaned.groupby('cluster')[features].mean()
print("Cluster Summary:")
print(cluster_summary)

# Save cluster summary to a CSV file
cluster_summary.to_csv('./data/cluster_summary.csv', index=True)

# Add cluster numbers to the original data
data_cleaned['cluster'] = clusters

# Convert cluster numbers to strings for mapping
data_cleaned['cluster_label'] = data_cleaned['cluster'].astype(str).map(cluster_labels)

# Save the original data with clusters and labels to a new CSV file
data_cleaned.to_csv('./data/original_data_with_clusters.csv', index=False)
