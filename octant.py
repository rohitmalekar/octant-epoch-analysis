import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Set page title
st.title('Octant Grant Analytics 2024')
st.subheader('Powered by OSO')
st.caption('Last updated: 02-Jan-2025')

# Read CSV files from data folder
try:
    monthly_events_df = pd.read_csv('./data/monthly_events_by_project.csv')
    proj_collections_df = pd.read_csv('./data/octant_all_collections.csv')
    dev_count_by_epoch_df = pd.read_csv('./data/dev_count_by_epoch.csv')
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
month_to_epoch_mapping = {
    'octant-02': [1, 2, 3],
    'octant-03': [4, 5, 6],
    'octant-04': [7, 8, 9],
    'octant-05': [10, 11, 12]
}

# Define the mapping from quarter to epoch
quarter_to_epoch_mapping = {
    1: 'octant-02',
    2: 'octant-03',
    3: 'octant-04',
    4: 'octant-05'
}

# Define the start month for each epoch
epoch_start_months = {
    'octant-02': 1,  # January
    'octant-03': 4,  # April
    'octant-04': 7,  # July
    'octant-05': 10  # October
}

# Add a new column 'epoch' to the DataFrame using the mapping
dev_count_by_epoch_df['epoch'] = dev_count_by_epoch_df['quarter'].map(quarter_to_epoch_mapping)

# Convert month column to datetime for easier filtering
monthly_events_df['bucket_month'] = pd.to_datetime(monthly_events_df['bucket_month'])

# Aggregate the data by project_id, event_type, and bucket_month
monthly_events_df = monthly_events_df.groupby(['project_id', 'event_type', 'bucket_month']).agg({
    'amount': 'sum'
}).reset_index()

# Read and concatenate epoch CSV files for funding
epoch_funding = pd.concat(epoch_dfs, ignore_index=True)

# Group by 'to_project_name' and 'grant_pool_name' and sum 'amount'
epoch_funding = epoch_funding.groupby(['to_project_name', 'grant_pool_name']).agg({'amount': 'sum'}).reset_index()

# Map grant_pool_name values
epoch_funding['grant_pool_name'] = epoch_funding['grant_pool_name'].replace({
    'epoch_2': 'octant-02',
    'epoch_3': 'octant-03',
    'epoch_4': 'octant-04',
    'epoch_5': 'octant-05'
})

summary_container = st.container()
summary_container.markdown('Over 6 Epochs starting Epoch Zero, Octant has funded approximately $5 million to 62 projects. \
                           Discover the stories behind the 47 open-source projects funded in 2024.')
summary_container.markdown("""
                            Explore how contributors and communities shape our digital future with data-driven insights into funding, developer engagement, and ecosystem growth.
                            Dive in to uncover trends, compare project performance, and inform strategic decisions for maximizing impact.
                            - **Visualize Progress**: Explore detailed visualizations of developer contributions and funding efficiency.
                            - **Tailor Insights**: Customize metrics to dive deeper into what matters to you.
                            - **Unlock Opportunities**: Learn how impactful ecosystems are evolving and thriving.
                            Start exploring today to celebrate and support the future of open source innovation.
                            """)


# Create tabs for different sections of the analysis
tab1, tab2, tab3, tab4 = st.tabs(["Analysis across Epochs", "Top Projects by Epoch", "Project Trends", "Strategic Findings"])

with tab1:
    st.markdown("#### Analyzing Developer Productivity and Funding Distribution Across Epochs")
    st.markdown("Explore how project productivity, measured as event contributions per active developer, correlates with funding received across various epochs. This interactive scatter plot highlights differences in developer team sizes and their impact on funding efficiency.")

    # Initialize an empty list to store DataFrames 
    epoch_dataframes = [] # Code metrics based on project participation in funding epochs
    epoch_dataframes_all = [] # Code metrics based on project participation in all epochs

    # Iterate over each collection name and its corresponding months
    for collection_name, months in month_to_epoch_mapping.items():
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
    
    # Merge with dev_count_by_epoch_df to get active_dev_count
    selected_event_data = selected_event_data.merge(
        dev_count_by_epoch_df[['project_id', 'epoch', 'active_dev_count']],
        on=['project_id', 'epoch'],
        how='left'
    )
    
    # Calculate amount_per_dev
    selected_event_data['amount_per_dev'] = selected_event_data['amount'] / selected_event_data['active_dev_count']

    # Define bins and labels for project size
    bins = [0, 2, 10, 20, 50, float('inf')]
    labels = ['Solo Projects (1-2 devs)', 'Small Teams (3-10 devs)', 'Medium Teams (11-20 devs)', 'Large Teams (21-50 devs)', 'Very Large Teams ( > 50 devs)']
    
    # Add project_size column
    selected_event_data['project_size'] = pd.cut(
        selected_event_data['active_dev_count'],
        bins=bins,
        labels=labels,
        right=True
    )

    # Merge selected_event_data with epoch_funding on project_name and epoch
    selected_event_data = selected_event_data.merge(
        epoch_funding,
        left_on=['project_name', 'epoch'],
        right_on=['to_project_name', 'grant_pool_name'],
        how='left'
    )

    selected_event_data.rename(columns={'amount_x': 'metric_amount'}, inplace=True)
    selected_event_data.rename(columns={'amount_y': 'funding_amount'}, inplace=True)

    # Drop the redundant columns after merge
    selected_event_data = selected_event_data.drop(columns=['to_project_name', 'grant_pool_name'])

    #st.dataframe(selected_event_data)

    # Create a scatter plot with metric_amount on Y axis and funding_amount on X axis
    fig = px.scatter(
        selected_event_data,
        x='funding_amount',
        y='amount_per_dev',
        color='project_size',  # Color dots based on project_size
        color_discrete_sequence=px.colors.qualitative.Bold,  # Use a bright color sequence
        facet_col='epoch',  # Create facets for each epoch
        facet_col_wrap=2,   # Arrange facets in 2 columns
        title=f'Scatter Plot of {event_type} per Active Developer vs Funding Amount',
        labels={'amount_per_dev': f'{event_type} Per Active Developer', 'funding_amount': 'Funding Amount'},
        hover_data=['project_name'],  # Add project name to hover data
        log_x=True,
        log_y=True  # Set Y-axis to logarithmic scale
    )

    fig.update_layout(
        height=800,
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="bottom",
            y=-0.2,  # Adjust the y position to place it below the plot
            xanchor="center",
            x=0.5  # Center the legend horizontally
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

with tab2:

    st.markdown("#### Customized Project Rankings and Epoch-Wise Performance")
    st.markdown("Assign weights to your selected metrics and explore the top-performing projects across funding epochs. Choose to normalize scores by team size for a balanced comparison or view absolute scores for a broader perspective.")

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

    # Add a toggle for the user to choose normalization method
    use_active_dev_count = st.checkbox('Normalize with Active Developer Count', value=True)
    st.caption("Enable Normalization with Active Developer Count to compare project scores relative to team size for a fairer assessment. Disable it to view absolute scores, irrespective of team size.")


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
        for i, epoch in enumerate(month_to_epoch_mapping.keys()):
            epoch_data = all_epoch_data[all_epoch_data['epoch'] == epoch]

            # Merge to include active_dev_count
            epoch_data = epoch_data.merge(
                dev_count_by_epoch_df[['project_id', 'epoch', 'active_dev_count']],
                on=['project_id', 'epoch'],
                how='left'
            )
            
            for metric in selected_metrics:
                if use_active_dev_count:
                    # Calculate the per-developer metric
                    epoch_data['per_dev_amount'] = epoch_data['amount'] / epoch_data['active_dev_count']

                    # Find the maximum per-developer value for normalization
                    max_value = epoch_data[epoch_data['event_type'] == metric]['per_dev_amount'].max()

                    # Normalize the per-developer metric
                    if max_value > 0:
                        epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = (
                            epoch_data['per_dev_amount'] / max_value
                        )
                    else:
                        epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = 0
                else:
                    # Find the maximum value for normalization without active_dev_count
                    max_value = epoch_data[epoch_data['event_type'] == metric]['amount'].max()

                    # Normalize the metric without active_dev_count
                    if max_value > 0:
                        epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = (
                            epoch_data['amount'] / max_value
                        )
                    else:
                        epoch_data.loc[epoch_data['event_type'] == metric, 'normalized_amount'] = 0
            
            epoch_data['composite_score'] = 0
            

            for metric in selected_metrics:
                # Calculate composite score factoring in normalized active developers
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
    for collection_name, months in month_to_epoch_mapping.items():
        
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
        lambda row: [row['octant-02'], row['octant-03'], row['octant-04'], row['octant-05']],
        axis=1
    )

    # Rename columns
    #pivot_funding_table = pivot_funding_table.rename(columns={
    #    'epoch_2': 'octant-02',
    #    'epoch_3': 'octant-03',
    #    'epoch_4': 'octant-04',
    #    'epoch_5': 'octant-05'
    #})


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
    The clustering methodology uses various features such as star counts, forks, developer contributions, and activity over **the last 6 months** to group projects into meaningful categories. Below is a summary of these clusters and their corresponding mean values across key metrics.
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
        column_order=['3', '0', '2', '1'],
        column_config={
            '0': st.column_config.Column(label='Steady Builders', width=150),
            '1': st.column_config.Column(label='High-Traffic Ecosystems', width=150),
            '2': st.column_config.Column(label='Established Pillars', width=150),
            '3': st.column_config.Column(label='Emerging Pioneers', width=150),
        }
    )

    # Cluster Descriptions
    st.markdown("##### Cluster Descriptions")
    
    st.image("./images/emerging.png", caption="Emerging Pioneers")
    st.markdown("###### Emerging Pioneers")
    st.markdown("""
    - Low to moderate star and fork count, suggesting niche or early-stage projects.
    - Smaller developer and contributor count, but active recent engagement is visible with higher relative active developer count and contributor count.
    - Moderate commit count and Merged PRs.
    - More recent first commit date, showcasing their status as relatively new or growing projects.""")

    st.image("./images/steady.png", caption="Steady Builders")
    st.markdown("###### Steady Builders")
    st.markdown("""
    - Moderate star and fork count, reflecting a stable and engaged user base.
    - Developer count and contributor count are relatively consistent, indicating sustained long-term engagement.
    - Commit count and Merged PRs are moderate, showcasing steady development activity.
    - Count of Closed Issues indicates regular maintenance efforts.
    - Projects in this cluster have older first commit date, implying they are established and focused on long-term sustainability.""")

  
    st.image("./images/pillars_and_high_traffic.png", caption="Established Pillars and High-Traffic Ecosystems")
    st.markdown("###### Established Pillars")
    st.markdown("""
    - Extremely high star and fork count, reflecting significant visibility and popularity in the ecosystem.
    - Moderate to high developer and contributor count, focusing on maintaining quality and impact.
    - Strong recent activity, with high commit count and Merged PRs.
    - Typically have a long history, with earlier first commit date, indicating that they are mature projects with consistent growth over time.
    """)

    st.markdown("###### High-Traffic Ecosystems")
    st.markdown("""
    - Very high star and fork count, indicating widespread popularity and adoption.
    - Significant developer and contributor count, reflecting large and vibrant teams.
    - High active developer and contributor count, showcasing dynamic recent engagement.
    - High commit count and Merged PRs, indicating rapid development cycles and active maintenance.
    - These projects often have earlier first commit date, reflecting their mature and established status.""")
    