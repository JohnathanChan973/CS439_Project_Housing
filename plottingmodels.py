import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
from pathlib import Path

SQLITE_DB_URL = Path("zillow_stats.db")

class PlottingModels:
    def plot_housing_map(df : pd.DataFrame, col : pd.Series, cluster : bool=False, 
                         n_clusters : int=10, value_name : str=None, figsize : tuple=(12, 10)):
        """
        Plots housing data on a map of the continental US, with the option of clustering.
        
        Parameters:
            df (pd.DataFrame): Merged DataFrame containing housing data with Latitude and Longitude columns
            col (pd.Series): Column from the passed in df that will be plotted
            cluster (bool, default=False): If True, perform KMeans clustering on the data
            n_clusters (int, default=10): Number of clusters for KMeans (only used if cluster=True) 
            value_name (str, optional): Name of the value being mapped
            figsize (tuple, default=(12, 10)): Figure size for the plot
            
        Returns:
            fig, ax (matplotlib figure and axis objects)
        """
        # This func could be refactored to allow plotting by a certain state. 
        # Can add a bool param for state, and the df should have StateName, so can check if all rows have the same StateName and uses that.
        # Could then have it restrict to a view of the states borders based on the min and max lat and long to encapsulate general area.
        # Likely wouldn't perfectly capture every state, but should look fine. Could also expose any remaining misplaced locations.

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
            crs="EPSG:4326"  # WGS 84 — standard lat/lon
        )

        gdf["RequestedCol"] = col
        
        # Extract lat/lon explicitly (needed for clustering)
        gdf["Latitude"] = gdf.geometry.y
        gdf["Longitude"] = gdf.geometry.x

        # Perform clustering if requested
        if cluster:
            # Normalize and Cluster
            features = gdf[["Latitude", "Longitude", "RequestedCol"]]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            gdf["cluster"] = kmeans.fit_predict(X_scaled)
        
        # Sets up Cartopy map
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Adds Cartopy features
        ax.add_feature(cfeature.LAND, facecolor='white')
        ax.add_feature(cfeature.BORDERS, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        
        # Restricts view to continental US
        ax.set_extent([-130, -65, 24, 50], crs=ccrs.PlateCarree())
        
        if cluster:
            # Plot with clusters
            scatter = gdf.plot(
                ax=ax,
                column="cluster",
                cmap="tab10",           # Distinct colors for clusters
                legend=True,
                markersize=20,
                alpha=0.8,
                transform=ccrs.PlateCarree()
            )
            if value_name:
                new_title = f"KMeans Clustering of {value_name} by Location and Value"
            else:
                new_title = "KMeans Clustering of Housing Prices by Location and Value"
        else:
            # Calculate IQR bounds for better color scaling
            Q1 = gdf["RequestedCol"].quantile(0.25)
            Q3 = gdf["RequestedCol"].quantile(0.75)
            IQR = Q3 - Q1
            
            vmin = Q1 - 1.5 * IQR
            vmax = Q3 + 1.5 * IQR
            
            # Clamp values so outliers don't mess up the scale
            vmin = max(vmin, gdf["RequestedCol"].min())
            vmax = min(vmax, gdf["RequestedCol"].max())
            
            # Plot with continuous color
            scatter = gdf.plot(
                ax=ax,
                column="RequestedCol",
                cmap="viridis",
                legend=True,
                markersize=20,
                alpha=0.7,
                vmax=vmax,
                vmin=vmin,
                transform=ccrs.PlateCarree()
            )
            if value_name:
                new_title = value_name
            else:
                new_title = "Housing Prices"
        
        # Set title with date
        if df["Date"].max():
            ax.set_title(f"{new_title} as of {df["Date"].max()}")
        else:
            ax.set_title(new_title)
        
        # Add credit for OpenStreetMap
        fig.text(
            .2, 0.01,
            "Coordinate Data from OpenStreetMap", 
            fontsize=11, color='black',
            ha='right', va='bottom'
        )
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def elbow_method(df : pd.DataFrame, col : str):
        """
        Performs the elbow omethod on the requested column in df to determine optimal amount for clustering

        Parameters:
            df (pd.DataFrame): DataFrame with Latitude and Longitude, which will are features for clustering
            col (str): Name of a column in the df which will have its values used as a third feature in the clustering
        """
        if "Latitude" not in df.columns or "Longitude" not in df.columns:
            print("Missing coordinate data.")
            return
        
        x = df.loc[:, [col, "Latitude", "Longitude"]]
        wcss = []
        for i in range(1, 50):
            kmeans= KMeans(i)
            kmeans.fit(x)
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        number_clusters = range(1, 50)
        plt.plot(number_clusters, wcss)
        plt.title('Elbow')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')

    def plot_time_series(region_ids : list, time_series_table : str, location_table : str, 
                         value_column : str, id_column : str="RegionID", name_column : str="RegionName", date_column : str="Date",
                         figsize : tuple=(12, 6), title : str=None, y_label : str=None, db_path : str=SQLITE_DB_URL):
        """
        Plot multiple time series from different regions on the same chart.
        
        Parameters:
            region_ids (list): List of region IDs to plot
            time_series_table (str): Name of the table containing time series data
            location_table (str): Name of the table containing location data
            value_column (str): Name of the column containing values to plot on y axis
            id_column (str): Name of the column containing region IDs
            name_column (str): Name of the column containing region names
            date_column (str): Name of the column containing dates
            figsize (tuple, default=(12,6)): Figure size of the plot
            title (str, optional): Plot title (defaults to '{value_column} Over Time by Region' if not given)
            y_label (str, optional): Y-axis label (defaults to value_column if not given)
            db_path (str, default=SQLITE_DB_URL):

        Raises:
            ValueError: If the table does not exist or required columns are missing
        """
        # Create placeholders for SQL IN clause
        placeholders = ','.join(['?' for _ in region_ids])
        
        # Construct the SQL query dynamically
        query = f"""
            SELECT ts.{date_column}, ts.{value_column}, loc.{name_column}, loc.{id_column}
            FROM {time_series_table} ts
            JOIN {location_table} loc USING ({id_column})
            WHERE loc.{id_column} IN ({placeholders})
        """
        with sqlite3.connect(db_path) as conn:
            # Checks that the tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            
            for table in [time_series_table, location_table]:
                if time_series_table not in valid_tables or location_table not in valid_tables:
                    raise ValueError(f"'{table}' does not exist in the database")

                # Checks if the expected columns are in their expected tables
                if table == time_series_table:
                    cursor.execute(f"PRAGMA table_info({table});")
                    table_columns = {row[1] for row in cursor.fetchall()}
                    
                    for col in [id_column, date_column, value_column]:
                        if col not in table_columns:
                            raise ValueError(f"'{col}' does not exist in '{table}'")
                        
                elif table == location_table:
                    cursor.execute(f"PRAGMA table_info({table});")
                    table_columns = {row[1] for row in cursor.fetchall()}
                    
                    for col in [id_column, name_column]:
                        if col not in table_columns:
                            raise ValueError(f"'{col}' does not exist in '{table}'")
                
            all_data = pd.read_sql_query(query, conn, params=region_ids) # Fetch data
        all_data[date_column] = pd.to_datetime(all_data[date_column], errors="coerce")
        
        # Setup plot
        plt.figure(figsize=figsize)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)

        num_regions = len(region_ids)
        colors = cm.viridis(np.linspace(0, 1, num_regions))

        for i, region_id in enumerate(region_ids):
            region_data = all_data[all_data[id_column] == region_id]
            if region_data.empty:
                continue
            ts = region_data.set_index(date_column)[value_column]
            plt.plot(ts, color=colors[i], label=region_data[name_column].iloc[0])

        plt.xlabel('Date')
        plt.ylabel(y_label if y_label else value_column)
        plt.title(title if title else f'{value_column} Over Time by Region')
        if len(region_ids) <= 10:
            plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return