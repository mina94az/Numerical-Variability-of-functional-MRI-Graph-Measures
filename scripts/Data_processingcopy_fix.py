# Libraries
from nilearn import datasets
import pandas as pd
import nibabel as nib

from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
# from nilearn import plotting

#import seaborn as sns
# Network Libraries
import networkx as nx

#import dash
import os

from joblib import Parallel, delayed

# Load the Schaefer 2018 atlas with 100 ROIs and 7 networks
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7)
atlas_labels =atlas.labels[1:]
atlas_filename=atlas.maps

# Define the column names of subjects attributes
columns = [
    'subject', 
    'session',
    'repetition', 
    'acquisition', 
    'degree_centralities', 
    'betweenness_centralities', 
    'eigenvector_centralities', 
    'clustering_coefficients', 
    'avg_shortest_path_length', 
    'small_worldness', 
    'correlation_matrix'
]
#Generating time series and correlation matrix
def compute_timeseries(func, confounds_tsv, use_confounds=True):
    """
    Compute time series with optional confound regression
    
    Parameters:
    - func: path to functional image
    - confounds_tsv: path to confounds file
    - use_confounds: boolean, whether to use confounds or not
    """
    fmri_img = nib.load(func)
    
    # Initialize the NiftiLabelsMasker with some standard options for preprocessing
    masker = NiftiLabelsMasker(atlas_filename, standardize=True, smoothing_fwhm=6, memory='nilearn_cache', verbose=5)
    
    if use_confounds:
        confounds = pd.read_table(confounds_tsv).fillna(method='bfill')
        confounds_to_include = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confounds = confounds[confounds_to_include]
        time_series = masker.fit_transform(fmri_img, confounds=confounds)
    else:
        time_series = masker.fit_transform(fmri_img, confounds=None)
    
    # Plot the correlation matrix
    correlation_measure = ConnectivityMeasure(kind='correlation')
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    
    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)

    return time_series, correlation_matrix

##########################################################################################################################################################################################################################################
import numpy as np
import networkx as nx
# Build graph from correlation matrix
def building_graph(correlation_matrix, correlation_threshold=None, sparsity_threshold=None):
    if correlation_threshold is None and sparsity_threshold is None:
        raise ValueError("You must provide either a 'correlation_threshold' or a 'sparsity_threshold'.")
    if sparsity_threshold is not None and not (0 < sparsity_threshold <= 1):
        raise ValueError("'sparsity_threshold' must be a value between 0 and 1.")
    num_nodes = correlation_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    final_threshold = 0
    if sparsity_threshold is not None:
        print(f"Using sparsity threshold: {sparsity_threshold}")
        upper_tri_indices = np.triu_indices(num_nodes, k=1)
        abs_correlations = np.abs(correlation_matrix[upper_tri_indices])
        sorted_correlations = np.sort(abs_correlations)[::-1]
        num_edges_to_keep = int(sparsity_threshold * len(sorted_correlations))
        if num_edges_to_keep > 0:
            final_threshold = sorted_correlations[num_edges_to_keep - 1]
        else:
            final_threshold = np.max(abs_correlations) + 1.0
        print(f"This corresponds to a correlation cut-off of: {final_threshold:.4f}")
    else:
        print(f"Using fixed correlation threshold: {correlation_threshold}")
        final_threshold = correlation_threshold
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            correlation = correlation_matrix[i, j]
            if abs(correlation) >= final_threshold:
                G.add_edge(i, j, weight=correlation)
    print(f"✅ Graph constructed with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

##########################################################################################################################################################################################################################################
# Compute graph measures
def compute_graph_measures(G):
    randMetrics = {"C": [], "L": []}
    C = nx.average_clustering(G)
    L = nx.average_shortest_path_length(G)
    for nrand in range (1,51):
        # Create a random graph with the same number of nodes and edges
        random_G = nx.erdos_renyi_graph(n=len(G.nodes), p=nx.density(G), seed=42)
        randMetrics["C"].append(nx.average_clustering(random_G))
        randMetrics["L"].append(nx.average_shortest_path_length(random_G))
        # Compute clustering coefficient and characteristic path length for the random graph
    Cr = np.mean(randMetrics["C"])
    Lr = np.mean(randMetrics["L"])
    # Compute small-worldness
    small_worldness = (C / Cr) / (L / Lr)
    return {
        'degree_centralities': nx.degree_centrality(G),
        'betweenness_centralities': nx.betweenness_centrality(G),
        'eigenvector_centralities': nx.eigenvector_centrality(G),
        'clustering_coefficients': nx.clustering(G),
        'avg_shortest_path_length': nx.average_shortest_path_length(G),
        'small_worldness': small_worldness
    }

##########################################################################################################################################################################################################################################
# Load data and compute graph measures for each subject, session, repetition
def load_data_compute(base, subject, iter, Results_table, mode='ieee', sparsity_threshold=None, correlation_threshold=None, use_confounds=True):
    """
    Modified function to accept correlation threshold and confound usage parameters
    """
    # Define the base directory path for the subject
    subject_directory = f'{base}/fmriprep_{mode}_outputs/iter_{iter}/{subject}'
    
    # Temporary list to store new rows
    new_rows = []
    
    # Check for session directories (e.g., ses-1, ses-2)
    if os.path.exists(subject_directory):
        for session in os.listdir(subject_directory):
            if session.startswith('ses-'):
                ses = session.split('-')[1]  # Extract session number
                directory = os.path.join(subject_directory, session, 'func')
                
                if os.path.exists(directory):
                    for filename in os.listdir(directory):
                        if filename.endswith('MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'):
                            func = os.path.join(directory, filename)
                            ses = os.path.basename(filename).split('_')[1].split('-')[1]
                            acq = os.path.basename(filename).split('_')[3]  # Extract acquisition info
                            print(f"analysing {subject}, {ses} , rep-{iter}, {acq}")
                            confound_tsv = os.path.join(directory, f'{subject}_ses-{ses}_task-rest_{acq}_desc-confounds_timeseries.tsv') 
                            
                            # Check if the row already exists in Results_table
                            existing_row = Results_table[
                                (Results_table['subject'] == subject) &
                                (Results_table['session'] == ses) &
                                (Results_table['repetition'] == f'rep-{iter}') &
                                (Results_table['acquisition'] == acq)
                            ]
                            
                            # If the row doesn't exist, perform computation
                            if existing_row.empty:
                                try:
                                    timeser, cormat = compute_timeseries(func, confound_tsv, use_confounds=use_confounds)
                                    G = building_graph(cormat, correlation_threshold, sparsity_threshold)
                                    adj_G = nx.to_numpy_array(G, weight='weight')
                                    res = compute_graph_measures(G)
                                    print(f'Computation done for file {filename}')

                                except Exception as e:
                                    print(f"Error processing {subject}, {ses}, rep-{iter}, {acq}: {e}")
                                    continue
                                
                                # Create a dictionary with the computed values
                                sub_data = {
                                    'subject': subject,
                                    'session': ses,
                                    'repetition': f'rep-{iter}',
                                    'acquisition': acq,
                                    'degree_centralities': res['degree_centralities'],
                                    'betweenness_centralities': res['betweenness_centralities'],
                                    'eigenvector_centralities': res['eigenvector_centralities'],
                                    'clustering_coefficients': res['clustering_coefficients'],
                                    'avg_shortest_path_length': res['avg_shortest_path_length'],
                                    'small_worldness': res['small_worldness'],
                                    'correlation_matrix': cormat,
                                    'adj_G': adj_G
                                }
                                
                                # Append the new dictionary to the list
                                new_rows.append(sub_data)
                                print(f"analysing {subject}, {ses} , rep-{iter}, {acq} , done completely")

                            else:
                                print(f'Skipping computation for {subject} session {ses}, repetition {iter}, acquisition {acq} (already computed)')
                else:
                    print(f'No "func" directory found for {subject} session {session}')
    else:
        print(f'Subject directory not found: {subject_directory}')
    
    # Convert the list of new rows to a DataFrame and concatenate
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        Results_table = pd.concat([Results_table, new_rows_df], ignore_index=True)
    
    return Results_table

##########################################################################################################################################################################################################################################
def process_batch(batch_info, base_dir, correlation_thresholds, use_confounds_options):
    """
    Process a single batch with different threshold and confound settings
    
    Parameters:
    - batch_info: dict with 'name' and 'csv_file' keys
    - base_dir: base directory path
    - correlation_thresholds: list of correlation thresholds to test
    - use_confounds_options: list of boolean values for confound usage
    """
    batch_name = batch_info['name']
    csv_file = batch_info['csv_file']
    
    print(f"\n{'='*50}")
    print(f"Processing {batch_name}")
    print(f"{'='*50}")
    
    # Load batch data
    batch_df = pd.read_csv(os.path.join(base_dir, batch_name, csv_file))
    subjects = batch_df['Subject'].unique()
    
    # Base directory for this batch
    batch_base = os.path.join(base_dir, batch_name)
    iterations = list(range(1,11))  # List of iterations (1 to 10)
    
    # Process each combination of threshold and confound setting
    for correlation_threshold in correlation_thresholds:
        for use_confounds in use_confounds_options:
            
            # Create output directory and filename first to check if file exists
            conf_suffix = "WConf" if use_confounds else "NoConf"
            output_dir = os.path.join(base_dir, f"table_corr{correlation_threshold}")
            output_filename = f"Result_table{conf_suffix}_{batch_name.lower()}.pkl"
            output_path = os.path.join(output_dir, output_filename)
            
            # Check if file already exists
            if os.path.exists(output_path):
                print(f"⏭️  Skipping {batch_name} with threshold {correlation_threshold}, confounds: {use_confounds} - File already exists: {output_filename}")
                continue
            
            print(f"\n--- Processing {batch_name} with threshold {correlation_threshold}, confounds: {use_confounds} ---")
            
            # Create empty Results_table for this configuration
            Results_table = pd.DataFrame(columns=columns)
            
            # Parallel processing
            Results_table_list = Parallel(n_jobs=-1, verbose=10)(
                delayed(load_data_compute)(
                    batch_base, subject, iter_num, Results_table.copy(), 'MCA', sparsity_threshold=None,
                    correlation_threshold=correlation_threshold, 
                    
                    use_confounds=use_confounds
                ) 
                for subject in subjects 
                for iter_num in iterations
            )
            
            # Concatenate results into a single DataFrame
            Results_table = pd.concat(Results_table_list, ignore_index=True)
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the Results_table to a pickle file
            Results_table.to_pickle(output_path)
            print(f"✅ Saved results to: {output_path}")
            print(f"   - Shape: {Results_table.shape}")

##########################################################################################################################################################################################################################################
def main():
    """
    Main function to process all batches
    """
    # Define base directory
    base_dir = '/home/mina94/links/projects/rrg-glatard/mina94/thesis_1/Data/extract_data'
    
    # Define batch information
    batch_configs = [
        {'name': 'Batch_1', 'csv_file': 'batch1_dataframe.csv'},
        {'name': 'Batch_2', 'csv_file': 'batch2_dataframe.csv'},
        {'name': 'Batch_3', 'csv_file': 'batch3_dataframe.csv'},
        {'name': 'Batch_4', 'csv_file': 'batch4_dataframe.csv'},
        {'name': 'Batch_5', 'csv_file': 'batch5_dataframe.csv'},
        {'name': 'Batch_hc', 'csv_file': 'batchhc_dataframe.csv'}
    ]
    
    # Define correlation thresholds to test (0.05 to 0.5 with 0.01 step)
    # correlation_thresholds = [round(x, 2) for x in np.arange(0.05, 0.51, 0.01)]
    correlation_thresholds=[0.05,0.1,0.2,0.3,0.4,0.5]
    # Define confound options
    use_confounds_options = [True, False]  # With and without confounds
    
    # Process each batch
    for batch_info in batch_configs:
        try:
            process_batch(batch_info, base_dir, correlation_thresholds, use_confounds_options)
        except Exception as e:
            print(f"❌ Error processing {batch_info['name']}: {e}")
            continue
    
    print("\n🎉 All batches processed successfully!")

# Run the main function
if __name__ == "__main__":
    main()