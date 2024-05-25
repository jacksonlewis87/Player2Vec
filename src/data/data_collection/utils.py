def row_to_header_dict(headers: list, row: list):
    if len(headers) != len(row):
        print(f"Error: header length != row length ({len(headers)}, {len(row)})")
        raise Exception

    return {headers[i]: row[i] for i in range(len(headers))}


def filter_dict_fields(_dict: dict, fields_to_include: list):
    return {key: value for key, value in _dict.items() if key in fields_to_include}


def sum_rows(list_of_rows: list, trim_start: int = 0, trim_end: int = 0):
    if len(list_of_rows) == 0:
        print("Error: empty list")
        raise Exception
    if trim_start + trim_end >= len(list_of_rows[0]):
        print("Error: trim amount greater than length of list")
        raise Exception
    return [
        sum([row[i + trim_start] for row in list_of_rows]) for i in range(len(list_of_rows[0]) - trim_start - trim_end)
    ]
