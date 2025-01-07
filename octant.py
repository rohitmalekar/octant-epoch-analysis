import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

# Set page title
st.title('Octant Grant Analytics 2024')
st.caption('Powered by OSO, Last updated: 06-Jan-2025')

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
                            """)

summary_container.markdown("""
                            It’s important to note that developer activity is an input metric rather than the desired impact outcome; however, it offers valuable context for understanding engagement and sustainability. Metrics like the number of active developers, the percentage of projects maintaining or increasing activity, and long-term growth trends illustrate Octant’s role in fostering a thriving ecosystem of public goods.
                            """)

# Create tabs for different sections of the analysis
tab1, tab2, tab3, tab4 = st.tabs(["Project Trends", "Analysis across Epochs", "Top Projects by Epoch", "Strategic Findings"])

with tab2:
    st.markdown("#### Analyzing Developer Productivity and Funding Distribution Across Epochs")
    st.markdown("Explore how project productivity, measured as event contributions per active developer, correlates with funding received across various epochs. This interactive scatter plot highlights differences in developer team sizes and their impact on funding efficiency.")
    st.markdown("""This section explores the question, **does funding correlate with developer productivity?** The analysis reveals that funding correlates only slightly with developer productivity, and this relationship is most evident in smaller teams. While higher funding can enhance contributions for small teams, the effect diminishes for medium and larger teams, where scaling often introduces complexities.
    """)

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
        
        # Ensure unique project_id and project_name pairs
        unique_projects = proj_collections_df[['project_id', 'project_name']].drop_duplicates()

        # Merge to include project_name
        epoch_data = epoch_data.merge(
            unique_projects,
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

with tab3:

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

with tab1:

    st.markdown("#### Project Trends: Insights Over Time")
    st.markdown("Analyzing project activity across multiple epochs uncovers valuable trends in contributions and funding patterns. By tracking key metrics like commits, PRs merged, and funding allocations over time, this section highlights how projects evolve and adapt.")

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

        # Ensure unique project_id and project_name pairs
        unique_projects = proj_collections_df[['project_id', 'project_name']].drop_duplicates()

        # Merge to include project_name
        epoch_data = epoch_data.merge(
            unique_projects,
            on='project_id',
            how='left'
        )
        
        # Append the DataFrame to the list
        epoch_dataframes.append(epoch_data)

    all_project_data = pd.concat(epoch_dataframes, ignore_index=True)

    all_project_data = all_project_data.drop_duplicates(subset=['project_id', 'project_name', 'event_type', 'epoch'])


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


    # Ensure unique project_id and project_name pairs
    unique_projects = proj_collections_df[['project_id', 'project_name']].drop_duplicates()
    # Merge to include project_name
    dev_count_by_epoch_df = dev_count_by_epoch_df.merge(
        unique_projects,
        on='project_id',
        how='left'
    )

    # Ensure unique project_id and project_name pairs in dev_count_by_epoch_df
    unique_dev_counts = dev_count_by_epoch_df[['project_id', 'project_name', 'epoch', 'active_dev_count']].drop_duplicates()

    # Merge epoch_funding with dev_count_by_epoch_df to include active_dev_count
    epoch_funding = epoch_funding.merge(
        unique_dev_counts[['project_name', 'epoch', 'active_dev_count']],
        left_on=['to_project_name', 'grant_pool_name'],
        right_on=['project_name', 'epoch'],
        how='left'
    )

    # Drop the redundant 'project_name' column from the merge
    epoch_funding = epoch_funding.drop(columns=['project_name'])    


    # Aggregate the data by project and epoch
    aggregated_funding = epoch_funding.groupby(['to_project_name', 'grant_pool_name']).agg({
        'amount': 'sum',
        'active_dev_count': 'sum'
    }).reset_index()

    # Reshape the aggregated data to have 'dev' and 'amount' as rows
    reshaped_funding_table = aggregated_funding.melt(
        id_vars=['to_project_name', 'grant_pool_name'],
        value_vars=['amount', 'active_dev_count'],
        var_name='metric',
        value_name='value'
    )

    # Pivot the reshaped table to have epochs as columns
    reshaped_funding_table = reshaped_funding_table.pivot_table(
        index=['to_project_name', 'metric'],
        columns='grant_pool_name',
        values='value',
        fill_value=0
    ).reset_index()

    # Add a trend column using LineChartColumn
    reshaped_funding_table['trend'] = reshaped_funding_table.apply(
        lambda row: [row['octant-02'], row['octant-03'], row['octant-04'], row['octant-05']],
        axis=1
    )


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

        filtered_funding_table = reshaped_funding_table[reshaped_funding_table['to_project_name'] == selected_project]

        # Drop the 'to_project_name' column
        filtered_funding_table = filtered_funding_table.drop(columns=['to_project_name'])

        filtered_funding_table = filtered_funding_table.sort_values(by='metric', ascending=False)

        # Ensure the values are numeric before formatting
        filtered_funding_table.loc[filtered_funding_table['metric'] == 'amount', filtered_funding_table.columns[1:]] = \
            filtered_funding_table.loc[filtered_funding_table['metric'] == 'amount', filtered_funding_table.columns[1:]].applymap(
                lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
            )
        
        # Rename the metric values
        filtered_funding_table['metric'] = filtered_funding_table['metric'].replace({
            'amount': 'Funding Amount',
            'active_dev_count': '# of Active Devs'
        })


        # Display the reshaped table with the trend column
        st.dataframe(
            filtered_funding_table,
            column_config={
                'trend': st.column_config.BarChartColumn(
                    label='Trend',
                    width="small",
                    y_min=0
                )
            },
            use_container_width=True,
            hide_index=True
        )
        

        # Display the pivot table with the trend column
        st.dataframe(
            filtered_code_metrics_table,
            column_config={

                'trend': st.column_config.BarChartColumn(
                    label='Trend',
                    width="small",
                    #y_min = 0
                )
            },
            use_container_width=True,
            hide_index=True
        )

with tab4:
    # Load and display the transformed cluster summary
    try:
        cluster_summary_df = pd.read_csv('./data/cluster_summary.csv')
        projects_with_clusters = pd.read_csv('./data/original_data_with_clusters.csv')
        pca_results = pd.read_csv('./data/pca_results.csv')
    except FileNotFoundError as e:
        st.error(f"Failed to load cluster_summary.csv: {e}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("cluster_summary.csv or original_data_with_clusters.csv is empty")
        st.stop()

    st.markdown("#### Clustering Insights for Tailored Capital Allocation")
    st.caption("""
        Click on a cluster to see the projects in that cluster.
    """)

    # Define color mapping for specific cluster labels
    cluster_color_mapping = {
        'Early-Stage Projects': '#FFDFBA',
        'High-Engagement Projects': '#FFB3BA',
        'Niche Growing Projects': '#BAFFC9',
        'Actively Evolving Projects': '#BAE1FF',
        'Mature Ecosystem Leaders': '#D7BDE2'
    }

    # Insert sunburst chart with specific color mapping
    fig = px.sunburst(
        projects_with_clusters,
        path=['cluster_label', 'project_name'],
        color='cluster_label',
        color_discrete_map=cluster_color_mapping  # Use the mapping dictionary here
    )

        # Update layout for better visualization
    fig.update_layout(
        margin=dict(t=40, l=0, r=0, b=0),
        height=600
    )

    # Display the sunburst chart in Streamlit
    st.plotly_chart(fig)    

    st.markdown("""
        The following analysis clusters projects based on their overall popularity (stars and forks), community size (developers and contributors), and how actively the community engages with the project. For each active developer, we look at their typical contribution patterns like how many commits they make, pull requests they submit, and issues they close, along with how long the project has been around.
    """)

    st.markdown("""
        The analysis highlights the diversity of Octant’s cohort of projects, underscoring the need for funding strategies tailored to the unique characteristics of each cluster. As the ecosystem scales, a one-size-fits-all approach may fail to optimize impact. Instead, designing allocation mechanisms based on clusters can better meet the distinct needs of these groups.
    """)

    st.markdown("""
        **Early-Stage Projects**
        - *Key Characteristics:* Recent origins, small teams, limited visibility (low star and fork counts), but relatively active given their scale.
        - *Actionable Insights:*
          - Targeted micro-grants or seed funding to encourage initial development and growth
          - Programs to enhance visibility, such as mentorship and partnerships with established projects
          - Tools for streamlined onboarding

        **High-Engagement Projects**
        - *Key Characteristics:* High star and fork counts, large contributor base, consistent activity, and mature development processes.
        - *Actionable Insights:*
          - Provide strategic funding focused on maintenance, scalability, and innovation
          - Encourage participation in collaborative initiatives to strengthen ecosystem-wide integration
          - Offer resources for managing high community demand

        **Niche Growing Projects**
        - *Key Characteristics:* Steady, mid-sized developer and contributor base with moderate visibility (stars, forks). Focused but consistent development cycles.
        - *Actionable Insights:*
          - Sustained funding to support iterative development and stability
          - Encourage outreach programs to increase awareness and attract a broader user base
          - Facilitate networking opportunities to connect with complementary initiatives

        **Actively Evolving Projects**
        - *Key Characteristics:* Relatively recent projects with smaller but highly active teams. High activity per developer, particularly in issues and pull requests.
        - *Actionable Insights:*
          - Flexible funding structures to support dynamic growth and rapid iteration
          - Resources for enhancing contributor onboarding and documentation
          - Highlight success stories to inspire similar projects

        **Mature Ecosystem Leaders**
        - *Key Characteristics:* Long-standing projects with very high visibility and broad contributor engagement. High levels of activity across all metrics, reflecting their critical role in the ecosystem.
        - *Actionable Insights:*
          - Focus funding on maintaining stability, addressing technical debt, and ensuring security
          - Invest in scalability to handle increasing community demand
          - Leverage these projects as flagship examples for the ecosystem
    """)
    

    st.markdown("""
        **Strategic Takeaway**
        Octant’s highly diverse project ecosystem requires tailored funding and support strategies to maximize impact. By aligning allocation mechanisms with the unique characteristics of these clusters, Octant can foster growth, sustainability, and scalability across its ecosystem, ensuring resources are directed where they are most needed. 
    """)

    # Methodology Section
    st.markdown("##### Methodology")
    #st.image("./images/all_clusters.png", caption="Visualization of Clustering Results")
    st.markdown("""
                The clustering methodology employs K-means (k=5) on MinMax-scaled features, with log transformation applied to handle highly skewed distributions in metrics like stars, forks, and per-developer activity rates. The high-dimensional feature space is visualized through PCA (Principal Component Analysis) reduction to 2D, while the original clustering is performed on the full feature set including project scale metrics (stars, forks, developer counts), normalized activity metrics (commits/PRs/issues per developer), and project maturity indicators (first commit date).
                """)
    st.markdown("""
                Interacting with the Scatter Plot:
                - **Highlight an area**: Click and drag on the scatter plot to zoom in on a specific region.
                - **Zoom out**: Click anywhere on the plot to reset and zoom out.
                """)

    # Define darker equivalents of the pastel colors
    darker_colors = ['#81C784', '#E57373', '#9B59B6', '#FFB74D', '#64B5F6']

    # Interactive Plotly scatter plot with project names displayed above each dot
    fig = px.scatter(
        pca_results,
        x='PCA1',
        y='PCA2',
        color='Cluster',
        #title="Clustering of Projects (Plotly Interactive)",
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
        height=800
    )

    fig.update_traces(marker=dict(size=10))
    
    # Display the sunburst chart in Streamlit
    st.plotly_chart(fig)
    
    #st.markdown("""
    #Below is a summary of these clusters and their corresponding mean values across key metrics.
    #""")

    
    #cluster_summary_df = cluster_summary_df.round(0)
    #transformed_cluster_summary_df = cluster_summary_df.set_index("cluster").T
    #if 'first_commit_date' in transformed_cluster_summary_df.index:
    #    transformed_cluster_summary_df.loc['first_commit_date'] = pd.to_datetime(
    #        transformed_cluster_summary_df.loc['first_commit_date']
    #    ).dt.year.astype(str)
    #for index in transformed_cluster_summary_df.index:
    #    if index != 'first_commit_date':
    #        transformed_cluster_summary_df.loc[index] = transformed_cluster_summary_df.loc[index].astype(int)

    #st.dataframe(
    #    transformed_cluster_summary_df,
    #    #column_order=['3', '0', '2', '1','4'],
    #    column_config={
    #        '0': st.column_config.Column(label='Early-Stage Projects', width=150),
    #        '1': st.column_config.Column(label='High-Engagement Projects', width=150),
    #        '2': st.column_config.Column(label='Niche Growing Projects', width=150),
    #        '3': st.column_config.Column(label='Actively Evolving Projects', width=150),
    #        '4': st.column_config.Column(label='Mature Ecosystem Leaders', width=150),
    #    }
    #)