import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch # <--- Required for custom legend
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
SHAPE_FILE = 'data/natural_earth_110m/ne_110m_admin_0_countries.shp'
DATA_FILE = 'data/relations_2010_to_2025.csv'
N_CLUSTERS = 3

# --- 1. Data Preparation ---
df = pd.read_csv(DATA_FILE)

country_profile = df.groupby('Country1').agg({
    'AvgTone': 'mean',
    'EventCount': 'sum'
}).reset_index()

country_profile = country_profile[country_profile['EventCount'] > 50]

# --- 2. Build the Model ---
scaler = StandardScaler()
features = ['AvgTone']
X_scaled = scaler.fit_transform(country_profile[features])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
country_profile['raw_cluster'] = kmeans.fit_predict(X_scaled)

# --- 3. Dynamic Re-Labeling ---
# Sort clusters by Average Tone so we know which is which
cluster_means = country_profile.groupby('raw_cluster')['AvgTone'].mean().sort_values()

# Map the sorted indices to labels
label_mapping = {
    cluster_means.index[0]: 'Negative',
    cluster_means.index[1]: 'Neutral',
    cluster_means.index[2]: 'Positive'
}
country_profile['Cluster Label'] = country_profile['raw_cluster'].map(label_mapping)

# --- 4. Visualization ---
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)

# Load Map
world = gpd.read_file(SHAPE_FILE, encoding='utf-8')
world = world[['ADM0_A3', 'geometry', 'NAME']]
world.columns = ['iso_code', 'geometry', 'country_name']
world = world[world['iso_code'] != 'ATA']

# Merge Data
map_data = world.merge(country_profile, left_on='iso_code', right_on='Country1', how='left')

# Define Colorblind-Safe Colors
color_dict = {
    'Negative': '#d55e00',  # Vermilion (Orange/Red)
    'Neutral':  '#8a8a8a',  # Light Grey
    'Positive': '#0072b2'   # Blue
}

# Plot Base Layer (Background)
world.plot(ax=ax, color='#eeeeee', edgecolor='white')

# Plot Data Layers
for label in ['Negative', 'Neutral', 'Positive']:
    subset = map_data[map_data['Cluster Label'] == label]
    if not subset.empty:
        subset.plot(
            ax=ax,
            color=color_dict[label],
            edgecolor='white',
            linewidth=0.5
        )

# --- FIX: Manually Create Legend Handles ---
legend_elements = [
    Patch(facecolor=color_dict['Positive'], edgecolor='white', label='Positive'),
    Patch(facecolor=color_dict['Neutral'], edgecolor='white', label='Neutral'),
    Patch(facecolor=color_dict['Negative'], edgecolor='white', label='Negative')
]

ax.legend(handles=legend_elements, loc='lower left', title='Average Tone', frameon=True)
ax.set_title('Global Geopolitical Tone (Clustered)', fontsize=16)
ax.set_axis_off()

plt.tight_layout()
plt.show()

# --- 5. Print Interpretation ---
print("Cluster Centroids (Sorted by Tone):")
print(cluster_means)