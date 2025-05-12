import pandas as pd
import sqlite3
import json
from pathlib import Path

DATA_DIR = Path("data")
SQLITE_DB_URL = Path("zillow_stats.db")

class DBManager: # All these functions access the db using the path, so may as well make a class
    def __init__(self, db_path=SQLITE_DB_URL):
        """
        Initializes using the given DB URL

        Parameters:
            db_path (str, default=SQLITE_DB_URL): Path to the SQLite database
        """
        self.db_path = db_path
        
    def table_to_df(self, table_name : str, cols : list=None):
        """
        Load data from a SQLite table into a pandas DataFrame
        
        Parameters:
            table_name (str): Name of the table to query
            cols (list, optional): Specific columns to select (defaults to all columns)
            
        Returns:
            pd.DataFrame: A DataFrame containing the queried data
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check that the table exists
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            
            if table_name not in valid_tables:
                raise ValueError(f"Table '{table_name}' does not exist in the database")
            
            # If cols is specified, check that all of them exist in the table
            if cols:
                cursor.execute(f"PRAGMA table_info({table_name});")
                table_columns = {row[1] for row in cursor.fetchall()}
                
                invalid_cols = [col for col in cols if col not in table_columns]
                if invalid_cols:
                    raise ValueError(f"Invalid column(s) for table '{table_name}': {', '.join(invalid_cols)}")
                
                columns_str = ", ".join(cols)
                query = f"SELECT {columns_str} FROM {table_name}" # Creates the query with the specified columns
            else:
                query = f"SELECT * FROM {table_name}" # No cols specified, so selects them all
            
            return pd.read_sql_query(query, conn)

    def select_date_table(self, table_name : str, cols : list=None, date : str="newest"):
        """
        Load data from a SQLite table with dates into a pandas DataFrame
        
        Parameters:
            table_name (str): Name of the table to query
            cols (list, optional): Specific columns to select (defaults to all columns)
            date (str, optional, default="newest"): "newest", "oldest", or a specific date string to filter by
            
        Returns:
            pd.DataFrame: A DataFrame containing the queried data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Checks that the table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            
            if table_name not in valid_tables:
                raise ValueError(f"Table '{table_name}' does not exist in the database")
            
            # Gets all columns from the table
            cursor.execute(f"PRAGMA table_info({table_name});")
            table_columns = {row[1] for row in cursor.fetchall()}
            
            # Checks that the Date column exists in the table
            if "Date" not in table_columns:
                raise ValueError(f"Invalid table. There is no 'Date' column in '{table_name}'")
            
            # Handles date filtering
            date_condition = ""
            if date == "newest":
                date_query = f"SELECT MAX(Date) FROM {table_name}"
                cursor.execute(date_query)
                max_date = cursor.fetchone()[0]
                date_condition = f"WHERE Date = '{max_date}'"
            elif date == "oldest":
                date_query = f"SELECT MIN(Date) FROM {table_name}"
                cursor.execute(date_query)
                min_date = cursor.fetchone()[0]
                date_condition = f"WHERE Date = '{min_date}'"
            elif date:
                date_condition = f"WHERE Date = '{date}'"
            
            # Handles column selection
            if cols:
                invalid_cols = [col for col in cols if col not in table_columns]
                if invalid_cols:
                    raise ValueError(f"Invalid column(s) for table '{table_name}': {', '.join(invalid_cols)}")
                
                columns_str = ", ".join(cols)
            else:
                columns_str = "*"
            
            # Creates the query
            query = f"""
                SELECT {columns_str}
                FROM {table_name}
                {date_condition}
            """
            
            return pd.read_sql_query(query, conn)
        
    def select_merged_table(self, table_names : str, key : str="RegionID", key_value : str=None, cols : list=None, date : str=None):
        """
        Load data from multiple SQLite tables merged on a common key with optional date filtering
        
        Parameters:
            table_names (set or list): Names of the tables to be merged and queried
            key (str, default="RegionID"): The column name to join tables on
            cols (list or tuple, optional): Specific columns to select (defaults to all columns)
            date (str, optional): "newest", "oldest", or a specific date string to filter by
            
        Returns:
            pd.DataFrame: A DataFrame containing the merged data
        """
        table_list = list(table_names) # Converts to list for easier indexing
        
        if len(table_list) != 2: # Merging more tables is tough and shouldn't be necessary for the project
            raise ValueError("Can only merge exactly two tables.")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Checks that all tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            
            invalid_tables = [table for table in table_list if table not in valid_tables]
            if invalid_tables:
                raise ValueError(f"Table(s) '{', '.join(invalid_tables)}' do not exist in the database")
            
            # Finds the table with the Date column so it can be filtered properly
            date_table = None
            for table in table_list:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = {row[1] for row in cursor.fetchall()}
                if "Date" in columns:
                    date_table = table
                    break
            
            if date_table is None:
                raise ValueError("None of the specified tables have a 'Date' column")
            
            # Get all columns from all tables to check against the requested columns
            all_columns = {}
            for table in table_list:
                cursor.execute(f"PRAGMA table_info({table});")
                all_columns[table] = {row[1] for row in cursor.fetchall()}
            
            # Validate the join key exists in both tables
            for table in table_list:
                if key not in all_columns[table]:
                    raise ValueError(f"Join key '{key}' not found in table '{table}'")
            
            # Handle date filtering
            date_condition = ""
            if date == "newest":
                date_query = f"SELECT MAX(Date) FROM {date_table}"
                cursor.execute(date_query)
                max_date = cursor.fetchone()[0]
                date_condition = f"{date_table}.Date = '{max_date}'"
            elif date == "oldest":
                date_query = f"SELECT MIN(Date) FROM {date_table}"
                cursor.execute(date_query)
                min_date = cursor.fetchone()[0]
                date_condition = f"{date_table}.Date = '{min_date}'"
            elif date:
                date_condition = f"{date_table}.Date = '{date}'"
            
            # Handle column selection
            if cols:
                # Check if specified columns exist in any of the tables
                available_cols = set()
                for table_cols in all_columns.values():
                    available_cols.update(table_cols)
                
                invalid_cols = [col for col in cols if col not in available_cols]
                if invalid_cols:
                    raise ValueError(f"Invalid column(s): {', '.join(invalid_cols)}")
                
                # Build column selection with table prefixes to avoid ambiguity
                select_cols = []
                for col in cols:
                    # Find which table(s) have this column
                    tables_with_col = [t for t in table_list if col in all_columns[t]]
                    if len(tables_with_col) == 1:
                        select_cols.append(f"{tables_with_col[0]}.{col}")
                    elif col == key:  # Join key appears in both tables, pick one
                        select_cols.append(f"{table_list[0]}.{col}")
                    else:  # Column appears in multiple tables, include both with aliases
                        for table in tables_with_col:
                            select_cols.append(f"{table}.{col} AS {table}_{col}")
                
                columns_str = ", ".join(select_cols)
            else:
                # Select all columns
                columns_str = "*"

            # Prepare key filter
            where_condition = ""
            if key_value:
                where_condition = f"{key} = {key_value}"

            # Combine conditions (instead of using multiple WHEREs)
            conditions = " AND ".join(filter(None, [date_condition, where_condition]))
            where_clause = f"WHERE {conditions}" if conditions else ""

            # Build the query
            query = f"""
                SELECT {columns_str}
                FROM {table_list[0]}
                JOIN {table_list[1]} USING ({key})
                {where_clause}
            """
            
            return pd.read_sql_query(query, conn)
        
    def add_to_table(self, table_name : str, df : pd.DataFrame):
        """
        Add or update records in a specified table using data from a DataFrame.
        
        Parameters:
            table_name (str): Name of the target table
            df (pd.DataFrame): DataFrame containing data to be added
        
        Return:
            int: Number of rows added
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Checks if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            if table_name not in valid_tables:
                raise ValueError(f"Invalid table name: {table_name}")
            
            # Gets column information
            cursor.execute(f"PRAGMA table_info({table_name});")
            table_columns = [row[1] for row in cursor.fetchall()]
            
            # Finds matching columns between the df and table
            df_columns = df.columns
            common_columns = [col for col in df_columns if col in table_columns]
            
            if not common_columns:
                raise ValueError(f"No matching columns found between given df and {table_name}")
            
            # Creates placeholders for the SQL query
            placeholders = ", ".join(["?" for _ in common_columns])
            columns_str = ", ".join(common_columns)
            
            # Creates INSERT statement to handle both new records
            insert_sql = f"""
                INSERT INTO {table_name} ({columns_str})
                VALUES ({placeholders})
            """
            
            # Prepares data for insertion
            insert_data = [
                tuple(row[col] for col in common_columns)
                for _, row in df.iterrows()
            ]
            
            # Execute and commit
            cursor.executemany(insert_sql, insert_data)
            conn.commit()
            
            return cursor.rowcount
    # Way used:
    # add_to_table("Metro_Location_Data", metro_loc_sales_raw)
    # continental_sales_raw_df[continental_sales_raw_df["RegionID"] == 753924]
    # continental_sales_raw_df.loc[continental_sales_raw_df["RegionID"] == 394437, "RegionName"] = "Canon City, CO"
    # continental_sales_raw_df.loc[continental_sales_raw_df["RegionID"] == 394437, "Latitude"] = 38.4422506
    # continental_sales_raw_df.loc[continental_sales_raw_df["RegionID"] == 394437, "Longitude"] = -105.2348795
    # continental_sales_raw_df[continental_sales_raw_df["RegionID"] == 394437]
    # add_to_table("Metro_Location_Data", continental_sales_raw_df[continental_sales_raw_df["RegionID"] == 394437])

    def safe_to_sqlite(self, df: pd.DataFrame, table_name: str):
        """
        Safely converts DataFrame into a SQLite table
            
        Parameters:
            df (pd.DataFrame): The DataFrame that is being added to the db
            table_name (str): The name that the table with have after it is added
        """
        with sqlite3.connect(self.db_path) as conn: # Should create the db if it does not already exist
            # Checks if table already exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,)
            )
            exists = cursor.fetchone() is not None

            if not exists:
                df.to_sql(table_name, conn, index=False)
                print(f"✅ Created '{table_name}'.")
            else:
                print(f"ℹ️ '{table_name}' already exists. Skipping...")

    def drop_column_sqlite(self, table_name : str, column_to_drop : str):
        """
        Drops a column from a table
            
        Parameters:
            table_name (str): The name of the table whose column is being removed
            column_to_drop (str): The name of the column that is being removed
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Gets columns from the table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            all_columns = [col[1] for col in columns_info]

            if column_to_drop not in all_columns:
                raise ValueError(f"Column '{column_to_drop}' does not exist in table '{table_name}'.")

            columns_to_keep = [col for col in all_columns if col != column_to_drop]
            columns_str = ', '.join(columns_to_keep)

            # Gets column definitions (name and type) for new table
            new_columns_def = ', '.join(
                f"{col[1]} {col[2]}" for col in columns_info if col[1] != column_to_drop
            )

            # Creates new table and puts data from old table in
            cursor.executescript(f"""
                CREATE TABLE new_{table_name} ({new_columns_def});

                INSERT INTO new_{table_name} ({columns_str})
                SELECT {columns_str} FROM {table_name};

                DROP TABLE {table_name};

                ALTER TABLE new_{table_name} RENAME TO {table_name};
            """)

            conn.commit()

    def update_datetime_column_to_date(self, table_name : str, datetime_column : str):
        """
        Changes a column in the given table from data in a datetime to a date format

        Parameters:
            table_name (str): The name of the table which needs its column adjusted
            datetime_column (str): The name of the column which needs to be adjusted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Updates column with datetime to only store the date (YYYY-MM-DD instead of YYYY-MM-DD hours:minutes:seconds:milliseconds) 
            update_query = f"""
            UPDATE {table_name}
            SET {datetime_column} = DATE({datetime_column});
            """
            cursor.execute(update_query)
            conn.commit()
            
    def rename_column_in_sqlite(self, table_name : str, old_column : str, new_column : str):
        """
        Changes the name of the given column in the given table to a new name

        Parameters:
            table_name (str): The name of the table which needs its column adjusted
            old_column (str): The name of the column which needs to be adjusted
            new_column (str): The new name of the column
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get the table's schema
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            if any(col[1] == old_column for col in columns): # Checks if column exists
                # Creates a new table with the updated column name
                columns_definition = ', '.join(
                    [f'"{col[1]}" {col[2]}' if col[1] != old_column else f'"{new_column}" {col[2]}'
                    for col in columns]
                )
                cursor.execute(f"""
                    CREATE TABLE new_{table_name} ({columns_definition});
                """)

                # Adds all the data into the new table from the old table
                cursor.execute(f"""
                    INSERT INTO new_{table_name} SELECT * FROM {table_name};
                """)

                # Drops the old table
                cursor.execute(f"DROP TABLE {table_name};")

                # Renames the new table to the old table's name
                cursor.execute(f"ALTER TABLE new_{table_name} RENAME TO {table_name};")

                print(f"Column {old_column} renamed to {new_column} successfully.")
            else:
                print(f"Column {old_column} does not exist in the table.")

            # Way used:
            # rename_column_in_sqlite('Metro_Sales_Data', 'Median Sales', 'Median_Sales')
            # rename_column_in_sqlite('Location_Data', 'State', 'StateName')

    def coordinates_to_json(self, table_name : str, output_dir: Path=DATA_DIR):
        """
        Extracts RegionID, Latitude, and Longitude from the given table and writes them to a JSON file

        Parameters:
            table_name (str): The name of the table to extract coordinate data from
            output_dir (str, default=DATA_DIR): Directory where the output JSON file will be saved

        Raises:
            ValueError: If the table does not exist or required columns are missing
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name=?;
            """, (table_name,))
            if not cursor.fetchone():
                raise ValueError(f"Table '{table_name}' does not exist in the database.")

            # Check for RegionID, Latitude, and Longitude columns since they will be the key and attributes in the json 
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = {col[1] for col in cursor.fetchall()}
            required = {"RegionID", "Latitude", "Longitude"}
            if not required.issubset(columns):
                raise ValueError(f"Table '{table_name}' must contain columns: {required}")

            # Fetch data
            cursor.execute(f"""
                SELECT RegionID, Latitude, Longitude FROM {table_name}
                WHERE Latitude IS NOT NULL AND Longitude IS NOT NULL;
            """)
            rows = cursor.fetchall()

            coord_dict = {
                str(region_id): {"Latitude": lat, "Longitude": lon}
                for region_id, lat, lon in rows
            }

            # Write to file
            output_dir = Path(output_dir) # Casts to a Path object because Python isn't strongly typed
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{table_name}.json"

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(coord_dict, f, indent=2)

            print(f"Coordinate data written to '{output_path}'")

    def update_table_coords(self, full_update_df : pd.DataFrame, table_name : str):
        """
        Updates a row's Latitude and Longitude in the table using the given df 

        Parameters:
            full_update_df (pd.DataFrame): df with columns named "Latitude-Update", "Longitude-Update", and "RegionID"
            table_name (str): The name of the table to extract coordinate data from

        Raises:
            ValueError: If the table does not exist
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Checks if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            valid_tables = {row[0] for row in cursor.fetchall()}
            if table_name not in valid_tables:
                raise ValueError(f"Invalid table name: {table_name}")

            update_sql = f"""
                UPDATE {table_name}
                SET Latitude = ?, Longitude = ?
                WHERE RegionID = ?
            """

            update_data = [ # Creates a list of tuples with the latitude, longitude, and regionid from the passed in df
                (row["Latitude-Update"], row["Longitude-Update"], row["RegionID"])
                for _, row in full_update_df.iterrows()
            ]

            cursor.executemany(update_sql, update_data) # Uses the list of tuples and updates the table with it
            conn.commit()
            
        # Way used:
        # weird_df = merged_df[(merged_df["Latitude"] > 50) & (merged_df["StateName"] != "AK")]
        # weird_df
        # for index, row in weird_df.iterrows():
        #     lat, long = get_city_coords(f"{row['RegionName']}, {row['City']}, California")
        #     if lat:
        #         weird_df.at[index, "Latitude"] = lat
        #     if long:
        #         weird_df.at[index, "Longitude"] = long
        # weird_df.loc[weird_df["RegionID"] == 810129, "Latitude"] = 34.036232, 
        # weird_df.loc[weird_df["RegionID"] == 810129, "Longitude"] = -117.613777
        # weird_df
        # update_table_coords(weird_df, "Metro_Location_Data")