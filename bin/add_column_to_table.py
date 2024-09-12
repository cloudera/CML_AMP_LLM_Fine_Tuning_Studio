import sqlite3
import argparse

def add_column_generic(db_path, old_table_name, new_column_name, new_column_type):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Step 1: Get the existing table schema
    cursor.execute(f"PRAGMA table_info({old_table_name});")
    columns_info = cursor.fetchall()
    
    # Step 2: Construct the new table schema
    columns = []
    for col in columns_info:
        columns.append(f"{col[1]} {col[2]}")  # col[1] is column name, col[2] is column type
    
    # Add the new column
    columns.append(f"{new_column_name} {new_column_type}")
    
    # Create the new table SQL
    columns_sql = ", ".join(columns)
    new_table_name = f"{old_table_name}_new"
    create_table_sql = f"CREATE TABLE {new_table_name} ({columns_sql});"
    
    # Step 3: Execute the creation of the new table
    cursor.execute(create_table_sql)
    
    # Step 4: Copy data from old table to new table
    existing_columns_list = ", ".join([col[1] for col in columns_info])
    cursor.execute(f"INSERT INTO {new_table_name} ({existing_columns_list}) SELECT {existing_columns_list} FROM {old_table_name};")
    
    # Step 5: Drop the old table
    cursor.execute(f"DROP TABLE {old_table_name};")
    
    # Step 6: Rename the new table to the old table name
    cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {old_table_name};")
    
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Add a new column to an SQLite table generically.")
    
    parser.add_argument("db_path", type=str, help="Path to the SQLite database file.")
    parser.add_argument("table_name", type=str, help="Name of the table to modify.")
    parser.add_argument("column_name", type=str, help="Name of the new column to add.")
    parser.add_argument("column_type", type=str, help="Data type of the new column (e.g., INTEGER, TEXT).")
    
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    add_column_generic(args.db_path, args.table_name, args.column_name, args.column_type)

if __name__ == "__main__":
    main()
