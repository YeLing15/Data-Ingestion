import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import pi
import argparse

def load_data(file_path):
    df = pd.read_csv(file_path)  # Load CSV file into a DataFrame
    
    df.replace('N/a', np.nan, inplace=True)  # Replace missing values denoted by 'N/a' with NaN

    exclude_cols = ['Player', 'Nation', 'Pos', 'Squad']  # Columns that should not be converted to numeric

    # Convert object columns to numeric, except the non-statistical columns
    for col in df.columns:
        if df[col].dtype == 'object' and col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Select numeric columns
    
    return df, numeric_cols

def perform_kmeans(df, stats_cols, important_cols):
    # Remove rows with NaN values in important columns
    df_filtered = df.dropna(subset=important_cols)

    # Standardize important columns for clustering
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_filtered[important_cols])

    # Determine optimal number of clusters using the elbow method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        inertia.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.savefig('./plot/kmeans.png', format='png', dpi=300)
    plt.show()

    # Use KMeans with optimal number of clusters (based on the elbow plot)
    optimal_k = 4  # Set manually after inspecting the elbow plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(df_scaled)

    return df_scaled  # Return scaled data for PCA

def pca_and_plot(df_scaled, num_clusters):
    # Perform PCA to reduce data to 2 dimensions
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df_scaled)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Plot the 2D PCA-transformed data
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis')
    plt.title(f'K-means Clustering with {num_clusters} clusters (PCA reduced)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig('./plot/pca.png', format='png', dpi=300)
    plt.show()

def radar_chart(df, player1, player2, attributes):
    # Extract data for both players
    p1_data = df[df['Player'] == player1][attributes].values.flatten()
    p2_data = df[df['Player'] == player2][attributes].values.flatten()

    # Create radar chart
    categories = attributes
    num_vars = len(categories)

    # Define angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    p1_data = np.append(p1_data, p1_data[0])  # Close plot
    p2_data = np.append(p2_data, p2_data[0])  # Close plot

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, p1_data, color='blue', alpha=0.25)
    ax.fill(angles, p2_data, color='red', alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.title(f'{player1} (Blue) vs {player2} (Red) Radar Chart', size=15, color='black', y=1.1)
    plt.savefig(f'./plot/radar_chart_{player1}_vs_{player2}.png', format='png', dpi=300)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two players using a radar chart.')
    parser.add_argument('--p1', type=str, required=True, help='First player name')
    parser.add_argument('--p2', type=str, required=True, help='Second player name')
    parser.add_argument('--Attribute', type=str, required=True, help='Comma-separated list of attributes to compare')
    args = parser.parse_args()

    player1 = args.p1
    player2 = args.p2
    attributes = args.Attribute.split(',')

    # Load the data
    df, stats_cols = load_data('./crawled/merged_premier_league_stats.csv')

    # Perform K-means clustering
    df_scaled = perform_kmeans(df, stats_cols, attributes)

    # Apply PCA and plot
    num_clusters = 4  # Set based on the elbow method
    pca_and_plot(df_scaled, num_clusters)

    # Radar chart comparison between players
    radar_chart(df, player1, player2, attributes)
