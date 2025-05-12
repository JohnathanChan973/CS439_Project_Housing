import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.distance import geodesic
import time

# This class was used for preprocessing to get all the coords, taking >10000 minutes (even longer with the failed attempts). Do not run again if not needed 
class Geocoder:
    def __init__(self, user_agent : str="zillow_housing_plotting_project", timeout : int=10, rate_limit_delay : int=1):
        """
        Initializes the geocoder
        
        Parameters:
            user_agent (str): User agent for Nominatim
            timeout (int): Timeout in seconds for geocoding requests
            rate_limit_delay (int): Delay between requests in seconds
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=timeout)
        self.rate_limit_delay = rate_limit_delay
        self.city_coords_cache = {}  # Cache for city coordinates
    
    def geocode(self, address : str, retry_count : int=0, max_retries : int=1):
        """
        Geocodes a given address
        
        Parameters:
            address (str): Address to geocode
            retry_count (int): Current retry attempt
            max_retries (int): Maximum number of retries
            
        Returns:
            tuple: (latitude, longitude) or (None, None) if geocoding fails
        """
        try:
            loc = self.geolocator.geocode(address)
            if loc:
                return loc.latitude, loc.longitude
            else:
                print(f"No results found for {address}")
                return None, None
                
        except GeocoderTimedOut:
            if retry_count < max_retries:
                print(f"Geocoding timed out for {address}, retrying... ({retry_count+1}/{max_retries})")
                time.sleep(2)
                return self.geocode(address, retry_count + 1, max_retries) # Calls itself recursively
            else:
                print(f"Max retries reached for {address}")
                return None, None
        except GeocoderUnavailable:
            print(f"Geocoding service unavailable for {address}")
            return None, None
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
            return None, None
    
    def geocode_metro(self, region_name : str):
        """
        Geocode a metro region name
        
        Parameters:
            region_name (str): Name of the metro region
            
        Returns:
            tuple: (latitude, longitude) or (None, None) if geocoding fails
        """
        return self.geocode(region_name)
    
    def geocode_region(self, location : str, state : str, city : str):
        """
        Geocode a region and uses the city level as a fallback if the region geocoding fails
        
        Parameters:
            location (str): Region/neighborhood name
            state (str): State name
            city (str): City name for fallback
            
        Returns:
            tuple: (latitude, longitude) or (None, None) if geocoding fails
        """
        # Try specific location first
        full_address = f"{location}, {state}"
        lat, lon = self.geocode(full_address)
        
        # Fallback to city level if needed
        if lat is None and location != city: # No need to check lon since if lat is None, both are None
            print(f"Falling back to city level for {full_address}")
            broad_address = f"{city}, {state}"
            lat, lon = self.geocode(broad_address)
            
        return lat, lon
    
    def add_coordinates_to_df(self, df : pd.DataFrame, region_col : str="RegionName", state_col : str="StateName", city_col : str=None, method : str="region"):
        """
        Add latitude and longitude coordinates to the given df
        
        Parameters:
            df (pd.DataFrame): DataFrame containing location data
            region_col (str): Name of column in df containing region names
            state_col (str): Name of column in df containing state names
            city_col (str, optional): Name of column in df containing city names 
            method (str): "metro", "region", or "city" - determines geocoding approach
            
        Returns:
            pd.DataFrame: DataFrame with added Latitude and Longitude columns
        """
        result_df = df.copy() # Creates a copy to avoid the df passed in
        
        # Lists to hold geocoded results
        latitudes = []
        longitudes = []
        
        print(f'Starting geocoding for {len(df)} locations using {method} method')
        
        # Process each row in the DataFrame
        for idx, row in df.iterrows():
            if idx % 50 == 0 and idx > 0:
                print(f"Processed {idx}/{len(df)} locations") # Just to keep track of how many are done
                
            if method == "metro":
                lat, lon = self.geocode_metro(row[region_col]) # Metro-level geocoding
            elif method == "region" and city_col:
                lat, lon = self.geocode_region(row[region_col], row[state_col], row[city_col]) # Region-level geocoding with city fallback
            elif method == "city":
                address = f"{row[city_col]}, {row[state_col]}"
                lat, lon = self.geocode(address) # City-level geocoding
            else:
                address = f"{row[region_col]}, {row[state_col]}"
                lat, lon = self.geocode(address) # Default geocoding approach
            
            latitudes.append(lat)
            longitudes.append(lon)
            
            time.sleep(self.rate_limit_delay) # Sleeps to avoid rate limits
        
        # Add coordinates to DataFrame
        result_df['Latitude'] = latitudes
        result_df['Longitude'] = longitudes
        
        print(f"Geocoding complete. Successfully geocoded {sum(lat is not None for lat in latitudes)} out of {len(latitudes)} locations")
        
        return result_df
    
    def build_city_coordinates_cache(self, df : pd.DataFrame, city_col : str="City", state_col : str="StateName"):
        """
        Build a cache of city center coordinates to minimize API calls
        
        Parameters:
            df (pd.DataFrame): DataFrame containing city and state data
            city_col (str): Name of column in df containing city names
            state_col (str): Name of column in df containing state names
            
        Returns:
            dict: Dictionary mapping "City, State" to (latitude, longitude)
        """
        # Create unique city-state combinations
        unique_addresses = df[[city_col, state_col]].drop_duplicates().dropna()
        print(f"Need to geocode {len(unique_addresses)} unique city-state combinations")
        
        # Geocode each unique city-state combination
        for _, row in unique_addresses.iterrows():
            address = f"{row[city_col]}, {row[state_col]}"
            if address not in self.city_coords_cache:  # Check if already in cache
                coords = self.geocode(address)
                if coords[0] is not None:
                    self.city_coords_cache[address] = coords
                time.sleep(self.rate_limit_delay)
        
        print(f"Successfully geocoded {len(self.city_coords_cache)} cities")
        return self.city_coords_cache
    
    def find_misplaced_regions(self, df : pd.DataFrame, distance_threshold : int=250, 
                              city_col : str="City", state_col : str="StateName", 
                              lat_col : str="Latitude", lon_col : str="Longitude", 
                              id_col : str="RegionID"):
        """
        Find regions that are too far from their city centers
        
        Parameters:
            df (pd.DataFrame): DataFrame with coordinates and city information
            distance_threshold (int): Maximum allowed distance in miles from city center
            city_col (str): Name of column in df containing city names
            state_col (str): Name of column in df containing state names
            lat_col (str): Name of column in df containing latitudes
            lon_col (str): Name of column in df containing longitudes
            id_col (str): Name of column in df containing region IDs
            
        Returns:
            dict: Dictionary mapping region IDs to corrected city center coordinates
        """
        # Ensure we have city coordinates
        if not self.city_coords_cache:
            self.build_city_coordinates_cache(df, city_col, state_col)
        
        # Find misplaced regions
        region_id_coords = {}
        count = 0
        
        for _, row in df.iterrows():
            # Skip rows with missing data
            if (pd.isna(row[city_col]) or pd.isna(row[state_col]) or 
                pd.isna(row[lat_col]) or pd.isna(row[lon_col])):
                continue
                
            address = f"{row[city_col]}, {row[state_col]}"
            if address not in self.city_coords_cache:
                continue  # Skip if city not in cache
                
            # Calculate distance from city center
            try:
                current_coords = (row[lat_col], row[lon_col])
                city_center = self.city_coords_cache[address]
                distance = geodesic(current_coords, city_center).miles
                
                if distance > distance_threshold:
                    region_id_coords[row[id_col]] = city_center
                    count += 1
            except ValueError:
                # Handle invalid coordinate values
                continue
        
        print(f"Found {count} regions more than {distance_threshold} miles from their city centers")
        return region_id_coords
    
    def correct_misplaced_regions(self, df : pd.DataFrame, misplaced_regions : dict, 
                                 id_col : str="RegionID", lat_col : str="Latitude", lon_col : str="Longitude"):
        """
        Correct coordinates for misplaced regions
        
        Parameters:
            df (pd.DataFrame): DataFrame with region coordinates
            misplaced_regions (dict): Dictionary mapping region IDs to corrected coordinates
            id_col (str): Name of columns in df containing region IDs
            lat_col (str): Name of columns in df containing latitudes
            lon_col (str): Name of columns in df containing longitudes
            
        Returns:
            pd.DataFrame: DataFrame with corrected coordinates
        """
        result_df = df.copy()
        
        for region_id, coords in misplaced_regions.items():
            mask = result_df[id_col] == region_id
            result_df.loc[mask, lat_col] = coords[0]
            result_df.loc[mask, lon_col] = coords[1]
        
        print(f"Corrected coordinates for {len(misplaced_regions)} regions")
        return result_df