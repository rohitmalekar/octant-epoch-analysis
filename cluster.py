# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from datetime import datetime

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
    "contributor_count_6_months",    
    "commit_count_6_months",
    "merged_pull_request_count_6_months",
    "closed_issue_count_6_months",
    "first_commit_date"
]


# Clean and preprocess date columns
data["first_commit_date"] = pd.to_datetime(data["first_commit_date"], errors="coerce")
data["last_commit_date"] = pd.to_datetime(data["last_commit_date"], errors="coerce")
data = data.dropna(subset=["first_commit_date", "last_commit_date"])
data["first_commit_date"] = data["first_commit_date"].apply(lambda x: x.toordinal())
data["last_commit_date"] = data["last_commit_date"].apply(lambda x: x.toordinal())


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
pca_df['Cluster'] = clusters
pca_df['project_name'] = data_cleaned['project_name']

# Ensure 'Cluster' is treated as a categorical variable
pca_df['Cluster'] = pca_df['Cluster'].astype(str)

# Define a custom color sequence with 5 dark contrasting colors
dark_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# Define custom labels for each cluster
cluster_labels = {
    '0': 'Emerging Projects',
    '1': 'Established Community Projects',
    '2': 'High-Traffic Collaborative Hubs',
    '3': 'Star Performers',
    '4': 'Specialized Focused Projects'
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
    color_discrete_sequence=dark_colors  # Use custom dark color sequence
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

# Add cluster labels to the original data
data['cluster'] = clusters

# Save the original data with clusters to a new CSV file
data.to_csv('./data/original_data_with_clusters.csv', index=False)