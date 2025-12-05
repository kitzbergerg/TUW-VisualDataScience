import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Point

# --- Configuration ---
DATA_FILE = 'data/country_goldstein.csv'
DEFAULT_ISO = 'AUT'
MIN_EVENTS = 50

# --- Load Data ---
world = gpd.read_file('data/natural_earth_110m/ne_110m_admin_0_countries.shp', encoding='utf-8').to_crs('EPSG:4326')
world.columns = world.columns.str.lower().str.replace(' ', '_')
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]

# Calculate global map bounds once to prevent camera jumping
MIN_X, MIN_Y, MAX_X, MAX_Y = world.total_bounds

# Load user interaction data
df = pd.read_csv(DATA_FILE)

# --- FIX 1: Symmetrize Data ---
# Create a copy with swapped source/target
df_rev = df.copy()
df_rev['f0_'] = df['f1_']
df_rev['f1_'] = df['f0_']

# Combine original and reversed, then group to handle duplicates
# This ensures A->B and B->A exist and have the same values
df = pd.concat([df, df_rev], ignore_index=True)
df = df.groupby(['f0_', 'f1_'], as_index=False).mean()


# --- Visualization Function ---
def plot_relationships(target_iso, ax):
    """Updates the map for the given target_iso."""
    ax.clear()
    ax.set_axis_off()

    # --- FIX 2: Set fixed bounds ---
    # Re-apply the global map limits after clearing
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)

    ax.set_title(f'Geopolitical Relations: Perspective of {target_iso}', fontsize=14)

    subset = df[df['f0_'] == target_iso].copy()
    merged = world.merge(subset, left_on='iso_a3', right_on='f1_', how='left')

    # 1. Base Layer
    world.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.5)

    # 2. Data Layer
    valid_data = merged[
        (merged['EventCount'] >= MIN_EVENTS) &
        (merged['AvgGoldstein'].notna())
        ]

    if not valid_data.empty:
        valid_data.plot(
            column='AvgGoldstein',
            ax=ax,
            cmap='RdYlGn',
            vmin=-10, vmax=10,
            edgecolor='white', linewidth=0.5
        )

    # Highlight source
    source_geo = world[world['iso_a3'] == target_iso]
    if not source_geo.empty:
        source_geo.plot(ax=ax, color='#377eb8', edgecolor='black', hatch='//')


# --- Interaction Logic ---
def on_click(event):
    if event.inaxes != ax: return

    point = Point(event.xdata, event.ydata)

    # Optimization: Filter potential hits by bounding box first (faster than iterating all)
    # or just use the loop as you had it (fine for small N)
    clicked_country = None
    for _, row in world.iterrows():
        if row['geometry'].contains(point):
            clicked_country = row['iso_a3']
            break

    if clicked_country:
        print(f"Clicked on {clicked_country}. Updating map...")
        plot_relationships(clicked_country, ax)
        fig.canvas.draw()


# --- Main Execution ---
fig, ax = plt.subplots(figsize=(14, 8))

norm = Normalize(vmin=-10, vmax=10)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdYlGn'), ax=ax, shrink=0.6)
cbar.set_label('Avg Goldstein Scale (-10: Enemy, +10: Ally)')

plot_relationships(DEFAULT_ISO, ax)

fig.canvas.mpl_connect('button_press_event', on_click)

plt.show()