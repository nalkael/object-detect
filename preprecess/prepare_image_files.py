import sqlite3


def create_tables():
    sql_statements = [
        '''
        CREATE TABLE IF NOT EXISTS orthomosaics (
            image_id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            image_name TEXT NOT NULL
        );
        '''
    ]

    # create a database connection
    try:
        with sqlite3.connect('orthomosaics.db') as conn:
            cursor = conn.cursor()
            for statement in sql_statements:
                cursor.execute(statement)
            conn.commit()
    except sqlite3.Error as e:
        print(e)


def insert_images():
    sql_statements = [
        """
        INSERT INTO orthomosaics ()
        """,
        """
        """,
        """
        """,
        """
        """,
        """
        """
    ]


if __name__ == '__main__':
    create_tables()
