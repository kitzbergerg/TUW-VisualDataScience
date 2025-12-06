import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import Slider, RadioButtons
from shapely.geometry import Point

# --- Configuration ---
DATA_FILE = 'data/relations_2010_to_2025.csv'
DEFAULT_ISO = 'UKR'
MIN_EVENTS = 50

# --- Load Data ---
world = gpd.read_file('data/natural_earth_110m/ne_110m_admin_0_countries.shp', encoding='utf-8').to_crs('EPSG:4326')
world.columns = world.columns.str.lower().str.replace(' ', '_')
world = world[(world.pop_est > 0) & (world.name != "Antarctica")]

# Calculate global map bounds once to prevent camera jumping
MIN_X, MIN_Y, MAX_X, MAX_Y = world.total_bounds

# Load user interaction data
df = pd.read_csv(DATA_FILE)

# Data symmetry
df_rev = df.copy()
df_rev['Country1'] = df['Country2']
df_rev['Country2'] = df['Country1']
df = pd.concat([df, df_rev], ignore_index=True)
df = df.groupby(['EventYear', 'Country1', 'Country2'], as_index=False).mean()

# Get available years
available_years = sorted(df['EventYear'].unique())
current_year = available_years[-1] # Default to most recent year
current_metric = 'AvgGoldstein'  # Default metric
current_iso = DEFAULT_ISO


def plot_relationships(target_iso, ax, year, metric):
    """Updates the map for the given target_iso, year range, and metric."""
    ax.clear()
    ax.set_axis_off()

    metric_name = 'Goldstein Scale' if metric == 'AvgGoldstein' else 'Tone'
    ax.set_title(f'Geopolitical Relations: {target_iso} ({year}) - {metric_name}', fontsize=14)

    subset = df[(df['Country1'] == target_iso) & (df['EventYear'] == year)].copy()
    merged = world.merge(subset, left_on='iso_a3', right_on='Country2', how='left')

    # 1. Base Layer
    world.plot(ax=ax, color='#d3d3d3', edgecolor='white', linewidth=0.5)

    # 2. Data Layer
    valid_data = merged[
        (merged['EventCount'] >= MIN_EVENTS) &
        (merged[metric].notna())
        ]

    if not valid_data.empty:
        # Set color scale based on metric
        if metric == 'AvgGoldstein':
            vmin, vmax = -10, 10
            cmap = 'RdYlGn'
        else:  # AvgTone
            vmin, vmax = -10, 10
            cmap = 'RdYlGn'

        valid_data.plot(
            column=metric,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            edgecolor='white', linewidth=0.5
        )

    # Highlight source
    source_geo = world[world['iso_a3'] == target_iso]
    if not source_geo.empty:
        source_geo.plot(ax=ax, color='#377eb8', edgecolor='black', hatch='//')

    # This prevents matplotlib from autoscaling based on the data
    ax.set_xlim(MIN_X, MAX_X)
    ax.set_ylim(MIN_Y, MAX_Y)
    ax.set_aspect('equal')


# --- Interaction Logic ---
def on_click(event):
    global current_iso

    if event.inaxes != ax:
        return

    point = Point(event.xdata, event.ydata)

    # Find clicked country
    clicked_country = None
    for _, row in world.iterrows():
        if row['geometry'].contains(point):
            clicked_country = row['iso_a3']
            break

    if clicked_country:
        print(f"Clicked on {clicked_country}. Updating map...")
        current_iso = clicked_country
        plot_relationships(current_iso, ax, current_year, current_metric)
        fig.canvas.draw_idle()


def on_year_change(val):
    global current_year
    current_year = int(val)
    plot_relationships(current_iso, ax, current_year, current_metric)
    fig.canvas.draw_idle()


def on_metric_change(label):
    global current_metric
    current_metric = 'AvgGoldstein' if label == 'Goldstein' else 'AvgTone'
    plot_relationships(current_iso, ax, current_year, current_metric)
    update_colorbar()
    fig.canvas.draw_idle()


def update_colorbar():
    """Update colorbar based on current metric."""
    cbar.ax.clear()

    if current_metric == 'AvgGoldstein':
        norm = Normalize(vmin=-10, vmax=10)
        label = 'Avg Goldstein Scale'
    else:
        norm = Normalize(vmin=-10, vmax=10)
        label = 'Avg Tone'

    sm = ScalarMappable(norm=norm, cmap='RdYlGn')
    fig.colorbar(sm, cax=cbar.ax)
    cbar.ax.set_ylabel(label)


if __name__ == '__main__':
    # Create figure with extra space for controls
    fig = plt.figure(figsize=(18, 9))

    # Main map axis
    ax = plt.axes([0.05, 0.05, 0.8, 0.9])

    # Colorbar
    cbar_ax = plt.axes([0.9, 0.05, 0.02, 0.9])
    norm = Normalize(vmin=-10, vmax=10)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='RdYlGn'), cax=cbar_ax)
    cbar.set_label('Avg Goldstein Scale')

    # Year range slider - positioned to avoid map overlap
    slider_ax = plt.axes([0.3, 0.05, 0.3, 0.03])
    year_slider = Slider(
        slider_ax,
        'Year Range',
        available_years[0],
        available_years[-1],
        valinit=current_year,
        valstep=1
    )
    year_slider.on_changed(on_year_change)

    # Metric radio buttons - positioned to avoid slider
    radio_ax = plt.axes([0.65, 0.035, 0.1, 0.06])
    radio_ax.set_title('Metric', fontsize=10)
    radio = RadioButtons(radio_ax, ('Goldstein', 'Tone'))
    radio.on_clicked(on_metric_change)

    # Initial plot
    plot_relationships(current_iso, ax, current_year, current_metric)

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()