import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Configuration ---
SHAPE_FILE = 'data/natural_earth_110m/ne_110m_admin_0_countries.shp'
DATA_FILE = 'data/relations_2010_to_2025.csv'
N_CLUSTERS = 3  # We assume ~5 major geopolitical blocs exists

# --- 1. Data Preparation (The Interaction Matrix) ---
df = pd.read_csv(DATA_FILE)

# We focus on the most recent trends (e.g., last 5 years) to get current blocs
df = df[df['EventYear'] >= 2020]

# Pivot the data:
# Rows = Who did the action (Country1)
# Cols = Who received the action (Country2)
# Values = The Tone of the action
interaction_matrix = df.pivot_table(
    index='Country1',
    columns='Country2',
    values='AvgGoldstein',
    aggfunc='mean'
).fillna(0) # Fill NaN with 0 (Neutral/No Interaction)

# Filter: Keep only countries that interact with at least 20 other countries
# This removes tiny islands that would just form "noise" clusters
active_countries = interaction_matrix.astype(bool).sum(axis=1)


# --- 2. Build the Model (PCA + K-Means) ---

# Step A: PCA (Dimensionality Reduction)
# We reduce the matrix to 10 "principal components" to remove noise
pca = PCA(n_components=10, random_state=42)
reduced_data = pca.fit_transform(interaction_matrix)

# Step B: K-Means Clustering on the reduced patterns
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
clusters = kmeans.fit_predict(reduced_data)

# Create a results dataframe
results = pd.DataFrame({
    'iso_a3': interaction_matrix.index,
    'Cluster': clusters
})

# --- 3. Visualization ---
world = gpd.read_file(SHAPE_FILE, encoding='utf-8')
world = world[['ADM0_A3', 'geometry', 'NAME']] # Keep only necessary columns
world.columns = ['iso_code', 'geometry', 'country_name']
world = world[world['iso_code'] != 'ATA'] # Exclude Antarctica

# Merge results
map_data = world.merge(results, left_on='iso_code', right_on='iso_a3', how='left')

# Plot
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_axis_off()
ax.set_title('Geopolitical Blocs (Detected via Interaction Patterns)', fontsize=16)

# Plot base world
world.plot(ax=ax, color='#e6e6e6', edgecolor='white')

# Plot clusters
valid_data = map_data.dropna(subset=['Cluster'])
valid_data.plot(
    column='Cluster',
    categorical=True,
    legend=True,
    cmap='tab10', # distinct colors for blocs
    linewidth=0.5,
    edgecolor='black',
    ax=ax,
    legend_kwds={'loc': 'lower left', 'title': 'Bloc ID'}
)

plt.show()

# --- 4. Interpretation ---
print("Bloc Members Analysis:")
for i in range(N_CLUSTERS):
    members = results[results['Cluster'] == i]['iso_a3'].tolist()
    print(f"\nCluster {i} ({len(members)} countries):")
    print(", ".join(members[:15]) + ("..." if len(members) > 15 else ""))