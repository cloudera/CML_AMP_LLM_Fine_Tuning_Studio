from ft.initialize_db import InitializeDB

def main():
    # Create an instance of InitializeDB
    db_initializer = InitializeDB()

    # Call the initialize_all method to initialize the database
    db_initializer.initialize_all()

if __name__ == "__main__":
    main()
