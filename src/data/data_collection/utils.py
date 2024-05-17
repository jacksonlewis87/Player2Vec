def row_to_header_dict(headers: list, row: list):
    if len(headers) != len(row):
        print(f"Error: header length != row length ({len(headers)}, {len(row)})")
        raise Exception

    return {headers[i]: row[i] for i in range(len(headers))}


def filter_dict_fields(_dict: dict, fields_to_include: list):
    return {key: value for key, value in _dict.items() if key in fields_to_include}
