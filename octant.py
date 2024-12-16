import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

# Set page title
st.title('Octant Grant Analytics - DRAFT')
st.subheader('Powered by OSO')

# Read CSV files from data folder
try:
    monthly_events_df = pd.read_csv('./data/monthly_events_by_project.csv')
    proj_collections_df = pd.read_csv('./data/octant_all_collections.csv')
except FileNotFoundError as e:
    raise Exception(f"Failed to load CSV files from data folder: {e}")
except pd.errors.EmptyDataError:
    raise Exception("One or more CSV files are empty")
except Exception as e:
    raise Exception(f"Error reading CSV files: {e}")

st.markdown('Over 5 Epochs, Octant has fundded $X to Y projects. The following analytics showcases Z of these projects that have open source contributions. In 2024, these projects have contributed to:')

# Convert month column to datetime for easier filtering
monthly_events_df['bucket_month'] = pd.to_datetime(monthly_events_df['bucket_month'])

monthly_events_df = monthly_events_df.groupby(['project_id', 'event_type', 'bucket_month']).agg({
    'amount': 'sum'
}).reset_index()

# Calculate total COMMIT_CODE amount for 2024
commit_code_2024 = monthly_events_df[
    (monthly_events_df['event_type'] == 'COMMIT_CODE') & 
    (monthly_events_df['bucket_month'].dt.year == 2024)
]['amount'].sum()

# Calculate total PULL_REQUEST_CLOSED amount for 2024
pull_request_closed_2024 = monthly_events_df[
    (monthly_events_df['event_type'] == 'PULL_REQUEST_CLOSED') & 
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
        label="Total Pull Requests Closed",
        value=int(pull_request_closed_2024),
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

    # Define a mapping of collection names to their corresponding months
    epoch_mapping = {
        'octant-02': [1, 2, 3],
        'octant-03': [4, 5, 6],
        'octant-04': [7, 8, 9],
        'octant-05': [10, 11, 12]
    }

    # Initialize an empty list to store DataFrames for each epoch
    epoch_dataframes = []

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
        ['COMMIT_CODE', 'PULL_REQUEST_CLOSED', 'ISSUE_CLOSED'],
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

    # Display the plot in Streamlit
    st.plotly_chart(fig)

with tab2:

    # Allow user to select up to three metrics
    selected_metrics = st.multiselect(
        'Select up to three metrics',
        ['COMMIT_CODE', 'PULL_REQUEST_CLOSED', 'ISSUE_CLOSED', 'FORKED', 'STARRED'],
        default=['COMMIT_CODE', 'PULL_REQUEST_CLOSED']
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

    # Define the start month for each epoch
    epoch_start_months = {
        'octant-02': 1,  # January
        'octant-03': 4,  # April
        'octant-04': 7,  # July
        'octant-05': 10  # October
    }

    # Create two columns
    col1, col2 = st.columns(2)

    # Add selectbox for epoch selection in the first column
    with col1:
        epoch = st.selectbox(
            'Select Epoch',
            ['octant-02', 'octant-03', 'octant-04', 'octant-05'],
            index=0  # Default to 'octant-02'
        )

    # Add selectbox for event type selection in the second column
    with col2:
        event_type = st.selectbox(
            'Select Event Type',
            ['COMMIT_CODE', 'PULL_REQUEST_CLOSED', 'ISSUE_CLOSED'],
            index=0  # Default to 'COMMIT_CODE'
        )

    # Filter the data for the selected epoch and event type
    filtered_data = monthly_events_df[
        (monthly_events_df['project_id'].isin(proj_collections_df[proj_collections_df['collection_name'] == epoch]['project_id'])) &
        (monthly_events_df['event_type'] == event_type) &
        (monthly_events_df['bucket_month'].dt.year == 2024) &  # Ensure the year is 2024
        (monthly_events_df['bucket_month'].dt.month >= epoch_start_months[epoch])
    ]


    # Ensure unique project_id and project_name pairs
    unique_projects = proj_collections_df[['project_id', 'project_name']].drop_duplicates()

    # Merge project names into the filtered data
    filtered_data = filtered_data.merge(
        unique_projects,
        on='project_id',
        how='left'
    )

    # Sort data by project and month
    filtered_data= filtered_data.sort_values(by=['project_name', 'bucket_month'])

    # Calculate cumulative sum for each project
    filtered_data['cumulative_amount'] = filtered_data.groupby('project_name')['amount'].cumsum()

    # Create a line chart for cumulative metrics with smooth curves
    fig_cumulative = px.line(
        filtered_data,
        x='bucket_month',
        y='cumulative_amount',
        color='project_name',
        title='Cumulative Metrics Over Time',
        labels={'cumulative_amount': 'Cumulative Total'},
        markers=True,
        line_shape='spline'
    )

    # Update layout to move legend to the bottom
    fig_cumulative.update_layout(
         showlegend=False, 
        height=800
    )

    # Display the cumulative line chart in Streamlit
    st.plotly_chart(fig_cumulative)

with tab4:
    st.header("Strategic Findings")
    # Add your strategic findings analysis code here
    # ...



