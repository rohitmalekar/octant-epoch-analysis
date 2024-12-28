import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Set page title
st.title('Octant Grant Analytics - DRAFT')
st.subheader('Powered by OSO')

# Read CSV files from data folder
try:
    monthly_events_df = pd.read_csv('./data/monthly_events_by_project.csv')
    monthly_active_devs_df = pd.read_csv('./data/monthly_active_devs_by_project.csv')
    proj_collections_df = pd.read_csv('./data/octant_all_collections.csv')
    # Read and concatenate epoch CSV files
    epoch_files = ['./data/epoch_2.csv', './data/epoch_3.csv', './data/epoch_4.csv', './data/epoch_5.csv']
    epoch_dfs = [pd.read_csv(file) for file in epoch_files]
except FileNotFoundError as e:
    raise Exception(f"Failed to load CSV files from data folder: {e}")
except pd.errors.EmptyDataError:
    raise Exception("One or more CSV files are empty")
except Exception as e:
    raise Exception(f"Error reading CSV files: {e}")

# Define a mapping of collection names to their corresponding months
epoch_mapping = {
    'octant-02': [1, 2, 3],
    'octant-03': [4, 5, 6],
    'octant-04': [7, 8, 9],
    'octant-05': [10, 11, 12]
}

# Define the start month for each epoch
epoch_start_months = {
    'octant-02': 1,  # January
    'octant-03': 4,  # April
    'octant-04': 7,  # July
    'octant-05': 10  # October
}


# Convert month column to datetime for easier filtering
monthly_events_df['bucket_month'] = pd.to_datetime(monthly_events_df['bucket_month'])

# Aggregate the data by project_id, event_type, and bucket_month
monthly_events_df = monthly_events_df.groupby(['project_id', 'event_type', 'bucket_month']).agg({
    'amount': 'sum'
}).reset_index()

# Convert month column to datetime for easier filtering
monthly_active_devs_df['bucket_month'] = pd.to_datetime(monthly_active_devs_df['bucket_month'])

# Aggregate the data by project_id, event_type, and bucket_month
monthly_active_devs_df = monthly_active_devs_df.groupby(['project_id', 'user_segment_type', 'bucket_month']).agg({
    'amount': 'sum'
}).reset_index()

# Read and concatenate epoch CSV files for funding
epoch_funding = pd.concat(epoch_dfs, ignore_index=True)

# Group by 'to_project_name' and 'grant_pool_name' and sum 'amount'
epoch_funding = epoch_funding.groupby(['to_project_name', 'grant_pool_name']).agg({'amount': 'sum'}).reset_index()


summary_container = st.container(border=True)
summary_container.markdown('Over 6 Epochs starting Epoch Zero, Octant has funded approximately $5 million to 62 projects. The following analytics showcases 47 of these projects that have open source contributions. In 2024, these projects have contributed to:')

with summary_container:
    # Calculate total COMMIT_CODE amount for 2024
    commit_code_2024 = monthly_events_df[
        (monthly_events_df['event_type'] == 'COMMIT_CODE') & 
        (monthly_events_df['bucket_month'].dt.year == 2024)
    ]['amount'].sum()

    # Calculate total PULL_REQUEST_CLOSED amount for 2024
    pull_request_merged_2024 = monthly_events_df[
        (monthly_events_df['event_type'] == 'PULL_REQUEST_MERGED') & 
        (monthly_events_df['bucket_month'].dt.year == 2024)
    ]['amount'].sum()

    # Calculate total ISSUE_CLOSED amount for 2024
    issue_closed_2024 = monthly_events_df[
        (monthly_events_df['event_type'] == 'ISSUE_CLOSED') & 
        (monthly_events_df['bucket_month'].dt.year == 2024)
    ]['amount'].sum()

    # Display metrics using Streamlit in three columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Total Commits",
            value=int(commit_code_2024),
            border=True
        )

    with col2:
        st.metric(
            label="Total Pull Requests Merged",
            value=int(pull_request_merged_2024),
            border=True
        )

    with col3:
        st.metric(
            label="Total Issues Closed",
            value=int(issue_closed_2024),
            border=True
        )

# Create tabs for different sections of the analysis
tab1, tab2, tab3, tab4 = st.tabs(["Analysis by Epoch", "Top Projects by Epoch", "Project Trends", "Strategic Findings"])

with tab1:
    st.markdown("#### Event Metrics Across Epochs")
    with st.expander("How to Interpret the Box Plot?"):
        st.markdown("""
            This box plot provides a visual summary of project activity for the selected event type (e.g., **COMMIT_CODE**, **PULL_REQUEST_CLOSED**, or **ISSUE_CLOSED**) across different epochs. Here's how to interpret it:

            - Each box represents the **range of activity levels** for projects within a specific epoch.
            - The **line inside the box** is the **median** (middle value).
            - The **ends of the box** are the **first quartile (25th percentile)** and **third quartile (75th percentile)**.
            - Points **outside the whiskers** indicate **outliers**—projects with unusually high or low activity compared to others.

            ---

            ##### What Can You Learn from This?

            - **Compare activity levels across epochs**: See which epochs had higher or lower overall activity for the selected event type.
            - **Identify outliers**: Hover over individual points to discover projects with extraordinary contributions, helping pinpoint leaders or anomalies in the ecosystem.
            - **Assess variability**: The size of the box and whiskers shows how spread out the activity levels are within an epoch.

            This analysis helps you understand trends, spotlight standout projects, and gauge how activity evolves over time.
            """)
        
    # Initialize an empty list to store DataFrames 
    epoch_dataframes = [] # Code metrics based on project participation in funding epochs
    epoch_dataframes_all = [] # Code metrics based on project participation in all epochs

    # Iterate over each collection name and its corresponding months
    for collection_name, months in epoch_mapping.items():
        # Filter projects for the current collection
        projects = proj_collections_df[proj_collections_df['collection_name'] == collection_name]
        
        # Get project IDs for filtering
        project_ids = projects['project_id'].unique()
        
        # Filter and aggregate metrics for the current epoch
        epoch_data = monthly_events_df[
            (monthly_events_df['project_id'].isin(project_ids)) &
            (monthly_events_df['bucket_month'].dt.year == 2024) &
            (monthly_events_df['bucket_month'].dt.month.isin(months))
        ].groupby(['project_id', 'event_type']).agg({
            'amount': 'sum'
        }).reset_index()
        
        # Add the epoch column
        epoch_data['epoch'] = collection_name
        
        # Merge to include project_name
        epoch_data = epoch_data.merge(
            projects[['project_id', 'project_name']],
            on='project_id',
            how='left'
        )
        
        # Append the DataFrame to the list
        epoch_dataframes.append(epoch_data)

    # Concatenate all epoch DataFrames into a single DataFrame
    all_epoch_data = pd.concat(epoch_dataframes, ignore_index=True)

    # Add a selectbox for event type selection with 'COMMIT_CODE' as the default
    event_type = st.radio(
        'Select Event Type',
        ['COMMIT_CODE', 'PULL_REQUEST_MERGED', 'ISSUE_CLOSED'],
        index=0  # Set the default index to 0 for 'COMMIT_CODE'
    )

    # Filter the data based on the selected event type
    selected_event_data = all_epoch_data[all_epoch_data['event_type'] == event_type]

    # Create a box plot with the selected event type
    fig = px.box(
        selected_event_data,
        x='epoch',
        y='amount',
        points='all',
        title=f'Box Plot of {event_type}',
        labels={'amount': 'Total'},
        log_y=True,  # Set y-axis to logarithmic scale
        hover_data=['project_name']  # Add project name to hover data
    )

    fig.update_layout(height=600) 

    # Display the plot in Streamlit
    st.plotly_chart(fig)

with tab2:

    # Allow user to select up to three metrics
    selected_metrics = st.multiselect(
        'Select up to three metrics',
        ['COMMIT_CODE', 'PULL_REQUEST_MERGED', 'ISSUE_CLOSED', 'FORKED', 'STARRED'],
        default=['COMMIT_CODE', 'PULL_REQUEST_MERGED']
    )

    # Initialize a dictionary to store weights
    weights = {}

    # Display sliders for each selected metric to assign weights
    for metric in selected_metrics:
        weights[metric] = st.slider(
            f'Weight for {metric}',
            min_value=0,
            max_value=100,
            value=50,  # Default value
            step=5
        )

    # Calculate total weight
    total_weight = sum(weights.values())

    # Display a button to show rankings if total weight is 100
    if total_weight == 100:
        st.info("Total weight is 100, showing rankings")
        
        # Create a new section for "Top Projects by Epoch"
        st.markdown("#### Top Projects by Epoch")
        
        # Create two columns for displaying tables
        col1, col2 = st.columns(2)
                
        # Initialize a dictionary to store project scores for each epoch
        epoch_project_scores = {}

        # Normalize metrics for each epoch
        for i, epoch in enumerate(epoch_mapping.keys()):
            epoch_data = all_epoch_data[all_epoch_data['epoch'] == epoch]
            
            # Normalize each selected metric
            for metric in selected_metrics:
                max_value = epoch_data[epoch_data['event_type'] == metric]['amount'].max()
                if max_value > 0:
                    epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = (
                        epoch_data['amount'] / max_value
                    )
                else:
                    epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = 0
            
            # Calculate composite score
            epoch_data['composite_score'] = 0
            for metric in selected_metrics:
                epoch_data.loc[epoch_data['event_type'] == metric, 'composite_score'] += (
                    epoch_data['normalized_amount'] * weights[metric] / 100
                )
            
            # Scale composite score to 0-100
            epoch_data['composite_score'] *= 100
            
            # Aggregate scores by project
            project_scores = epoch_data.groupby('project_id').agg({
                'composite_score': 'sum',
                'project_name': 'first'
            }).reset_index()
            
            # Save project_scores for the current epoch
            epoch_project_scores[epoch] = project_scores

            # Sort projects by composite score and select top 5
            top_projects = project_scores.sort_values(by='composite_score', ascending=False).head(5)
            
            # Display the top projects for the current epoch in the appropriate column
            with (col1 if i % 2 == 0 else col2):
                st.markdown(f"##### {epoch}")
                st.dataframe(
                    top_projects[['project_name', 'composite_score']],
                    column_config={
                        'composite_score': st.column_config.ProgressColumn(
                            label='Composite Score',
                            format="%d",
                            min_value=0,
                            max_value=100
                        )
                    },
                    use_container_width=True,  # Optional: to use the full width of the container
                    hide_index=True  # This will hide the index column
                )

        # After the loop, create a grid showing project versus epoch
        # Combine all project scores into a single DataFrame
        all_scores = pd.concat(epoch_project_scores, names=['epoch', 'index']).reset_index(level='epoch')

        # Pivot the DataFrame to create a grid
        project_epoch_grid = all_scores.pivot_table(
            index='project_name',
            columns='epoch',
            values='composite_score',
            #fill_value=0  # Fill missing values with 0
        )

        # Sort the grid based on the sum of scores across all columns
        project_epoch_grid['total_score'] = project_epoch_grid.sum(axis=1)
        project_epoch_grid = project_epoch_grid.sort_values(by='total_score', ascending=False)
        project_epoch_grid = project_epoch_grid.drop(columns='total_score')

        # Display the grid in Streamlit
        st.markdown("#### Project vs Epoch Grid")
        st.dataframe(project_epoch_grid, 
                     column_config={
                         'octant-02': st.column_config.ProgressColumn(
                             format="%d",
                             min_value=0,
                             max_value=100
                         ),
                         'octant-03': st.column_config.ProgressColumn(
                             format="%d",
                             min_value=0,
                             max_value=100
                         ),
                         'octant-04': st.column_config.ProgressColumn(
                             format="%d",
                             min_value=0,
                             max_value=100
                         ),
                         'octant-05': st.column_config.ProgressColumn(
                             format="%d",
                             min_value=0,
                             max_value=100
                         ),
                     },
                     use_container_width=True)

    else:
        st.warning("Total weight must be 100 to show rankings.")

with tab3:

    # Get project IDs for filtering
    project_ids = proj_collections_df['project_id'].unique()
        
    # Get code metrics for each project irrespective of they being part of a funding epoch
    for collection_name, months in epoch_mapping.items():
        
        # Filter and aggregate metrics for the current epoch
        epoch_data = monthly_events_df[
            (monthly_events_df['project_id'].isin(project_ids)) &
            (monthly_events_df['bucket_month'].dt.year == 2024) &
            (monthly_events_df['bucket_month'].dt.month.isin(months))
        ].groupby(['project_id', 'event_type']).agg({
            'amount': 'sum'
        }).reset_index()
        
        # Add the epoch column
        epoch_data['epoch'] = collection_name
        
        # Merge to include project_name
        epoch_data = epoch_data.merge(
            proj_collections_df[['project_id', 'project_name']],
            on='project_id',
            how='left'
        )
        
        # Append the DataFrame to the list
        epoch_dataframes.append(epoch_data)

    all_project_data = pd.concat(epoch_dataframes, ignore_index=True)

    # Create the pivot table with filtered data in one step
    pivot_code_metrics_table = all_project_data[
        all_project_data['event_type'].isin(['FORKED', 'STARRED', 'COMMIT_CODE', 'ISSUE_CLOSED', 'PULL_REQUEST_MERGED'])
    ].pivot_table(
        index=['project_name', 'event_type'],
        columns='epoch',
        values='amount',
        aggfunc='sum',  # Use sum to aggregate values if there are duplicates
        fill_value=0    # Fill missing values with 0
    ).reset_index()

    # Add a trend column using LineChartColumn
    pivot_code_metrics_table['trend'] = pivot_code_metrics_table.apply(
        lambda row: [row['octant-02'], row['octant-03'], row['octant-04'], row['octant-05']],
        axis=1
    )
    
    # Rename columns
    pivot_code_metrics_table = pivot_code_metrics_table.rename(columns={
        'octant-02': 'Jan - Mar',
        'octant-03': 'Apr - Jun',
        'octant-04': 'Jul - Sep',
        'octant-05': 'Oct - Dec'
    })


    pivot_funding_table = epoch_funding.pivot_table(
        index=['to_project_name'],
        columns='grant_pool_name',
        values='amount',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    pivot_funding_table['trend'] = pivot_funding_table.apply(
        lambda row: [row['epoch_2'], row['epoch_3'], row['epoch_4'], row['epoch_5']],
        axis=1
    )

    # Rename columns
    pivot_funding_table = pivot_funding_table.rename(columns={
        'epoch_2': 'octant-02',
        'epoch_3': 'octant-03',
        'epoch_4': 'octant-04',
        'epoch_5': 'octant-05'
    })


    # Format values with rounded numbers and dollar sign
    for col in ['octant-02', 'octant-03', 'octant-04', 'octant-05']:
        pivot_funding_table[col] = pivot_funding_table[col].apply(lambda x: f"${x:,.0f}")


    # Extract unique project names from all_epoch_data
    project_names = all_epoch_data['project_name'].unique()

    # Add a search box with auto-suggest for project names
    selected_project = st.selectbox(
        'Search for a project',
        options=project_names,
        format_func=lambda x: x if x else "Select a project"
    )

    # Display selected project details or perform actions based on selection
    if selected_project:
        
        # Add any additional logic to display project-specific data

        # Filter the pivot table for the selected project
        filtered_code_metrics_table = pivot_code_metrics_table[pivot_code_metrics_table['project_name'] == selected_project]
        # Drop project name column
        filtered_code_metrics_table = filtered_code_metrics_table.drop(columns=['project_name'])

        filtered_funding_table = pivot_funding_table[pivot_funding_table['to_project_name'] == selected_project]

        st.dataframe(
            filtered_funding_table,
            column_config={
                'trend': st.column_config.BarChartColumn(
                    label='Trend',
                    width=150,
                    y_min = 0
                )
            },
            use_container_width=True,
            hide_index=True
        )

        # Display the pivot table with the trend column
        st.dataframe(
            filtered_code_metrics_table,
            column_config={

                'trend': st.column_config.LineChartColumn(
                    label='Trend',
                    width=150,
                    y_min = 0
                )
            },
            use_container_width=True,
            hide_index=True
        )

with tab4:

    st.markdown("#### Clustering Insights for Tailored Capital Allocation in Octant V2")

    # Contextual Introduction
    with st.expander("📜 **Introduction**", expanded=True):
        st.markdown("""
        The [Octant V2 framework](https://octantapp.notion.site/Degens-Dragons-Octant-V2-Overview-full-version-0-4-127e165689aa8022bb01dfc76e3cca4d) envisions a dynamic and decentralized approach to deploy capital and efficiently stream value to impactful initiatives within a project's ecosystem. This is achieved by leveraging community-driven decision-making and transparent onchain mechanisms.

        In any large ecosystem, different projects require varying types of support depending on their maturity, popularity, and level of community engagement. Octant addresses these diverse needs while maximizing impact by embracing a **plurality of allocation strategies**.

        > *By categorizing projects into distinct clusters—ranging from emerging initiatives to high-traffic hubs and star performers—Octant can tailor allocation strategies to match the unique needs and potential of each group.*

        These clusters provide the foundation for understanding the underlying characteristics of funded projects and inform data-driven allocation strategies.
        """)

    # Methodology Section
    st.markdown("##### Methodology")
    st.image("./images/all_clusters.png", caption="Visualization of Clustering Results")

    st.markdown("""
    The clustering methodology uses various features such as star counts, forks, developer contributions, and activity over time to group projects into meaningful categories. Below is a summary of these clusters and their corresponding mean values across key metrics.
    """)

    # Load and display the transformed cluster summary
    try:
        cluster_summary_df = pd.read_csv('./data/cluster_summary.csv')
    except FileNotFoundError as e:
        st.error(f"Failed to load cluster_summary.csv: {e}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("cluster_summary.csv is empty")
        st.stop()

    cluster_summary_df = cluster_summary_df.round(0)
    transformed_cluster_summary_df = cluster_summary_df.set_index("cluster").T
    if 'first_commit_date' in transformed_cluster_summary_df.index:
        transformed_cluster_summary_df.loc['first_commit_date'] = pd.to_datetime(
            transformed_cluster_summary_df.loc['first_commit_date']
        ).dt.year.astype(str)
    for index in transformed_cluster_summary_df.index:
        if index != 'first_commit_date':
            transformed_cluster_summary_df.loc[index] = transformed_cluster_summary_df.loc[index].astype(int)

    st.dataframe(
        transformed_cluster_summary_df,
        column_order=['0', '4', '1', '2', '3'],
        column_config={
            '0': st.column_config.Column(label='Emerging Projects', width=150),
            '1': st.column_config.Column(label='Established Community Projects', width=150),
            '2': st.column_config.Column(label='High-Traffic Collaborative Hubs', width=150),
            '3': st.column_config.Column(label='Star Performers', width=150),
            '4': st.column_config.Column(label='Specialized Focused Projects', width=150)
        }
    )

    # Cluster Descriptions
    st.markdown("##### Cluster Descriptions")
    
    st.image("./images/emerging.png", caption="Emerging and Specialized Projects")
    st.markdown("###### Emerging Projects")
    st.markdown("""
    - Moderate star and fork counts with relatively few developers.
    - Active contributors over the past 6 months, indicating recent interest.
    - Lower overall activity compared to other clusters, suggesting a project in its growth phase.
    """)

    st.markdown("###### Specialized Focused Projects")
    st.markdown("""
    - Moderate star and fork counts with a small but dedicated developer and contributor base.
    - High commit and pull request activity relative to the team size, indicating intense work by a focused team.
    - Likely niche or highly specialized projects driven by committed contributors.
    """)
  
    st.image("./images/established.png", caption="Established, High-Traffic, and Star Performers")
    st.markdown("###### Established Community Projects")
    st.markdown("""
    - High star and fork counts with a strong contributor base.
    - Moderate recent developer activity, suggesting steady ongoing engagement.
    - Likely long-standing projects with consistent user interest and moderate growth.
    """)

    st.markdown("###### High-Traffic Collaborative Hubs")
    st.markdown("""
    - Very high contributor and developer counts, along with significant star and fork counts.
    - High activity in the past 6 months, including commits, pull requests, and issue resolution.
    - Indicates large, vibrant ecosystems with extensive collaboration and active community management.
    """)

    st.markdown("###### Star Performers")
    st.markdown("""
    - Exceptionally high star and fork counts, indicative of widespread popularity.
    - Moderate developer and contributor counts, with steady recent activity.
    - Likely mature, high-visibility projects with a stable but less dynamic contributor base.
    """)
    
