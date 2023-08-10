from typing import Any

import numpy as np
import pyodbc

from src import utils


class MsSql:
    def __init__(self, server: str, uid: str, pwd: str, database: str):
        connection_string = (
            "Driver={ODBC Driver 18 for SQL Server};"
            f"Server={server};"
            f"Database={database};"
            f"uid={uid};"
            f"pwd={pwd};"
            "TrustServerCertificate=yes;"
        )
        connection = pyodbc.connect(connection_string, autocommit=True)
        self.cursor = connection.cursor()

    def get_data(
        self, table_name: str, condition: dict[str, Any] | None = None
    ) -> tuple[utils.Features, utils.Target]:
        request_builder = [f"SELECT * FROM {table_name}"]
        if condition is not None:
            where_str = utils.get_condition_str(condition)
            request_builder.append(f" WHERE {where_str}")
        request_builder.append(";")
        request = "".join(request_builder)

        self.cursor.execute(request)
        data = self.cursor.fetchall()

        x, y = [], []
        for row in data:
            x.append(row[:-2])
            y.append(row[-2])
        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    def insert_row(self, table_name: str, row: tuple[Any, ...]):
        row_str = ", ".join(map(str, row))
        self.cursor.execute(f"""INSERT INTO {table_name} VALUES ({row_str});""")

    def update_row(
        self,
        table_name: str,
        update_condition: dict[str, Any],
        updated_values: dict[str, Any],
    ):
        set_str = ", ".join(
            [f"[{key}]={value}" for key, value in updated_values.items()]
        )
        where_str = utils.get_condition_str(update_condition)
        self.cursor.execute(f"""UPDATE {table_name} SET {set_str} WHERE {where_str}""")
