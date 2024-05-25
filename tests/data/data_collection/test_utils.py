import pytest

from data.data_collection.utils import row_to_header_dict, filter_dict_fields, sum_rows


@pytest.mark.parametrize(
    "row, raises_exception",
    [
        ([0, 1, 2], False),
        ([0, 1, 2, 3], True),
    ],
)
def test_row_to_header_dict(row, raises_exception):
    headers = ["zero", "one", "two"]

    if raises_exception:
        with pytest.raises(Exception):
            result = row_to_header_dict(headers=headers, row=row)
    else:
        result = row_to_header_dict(headers=headers, row=row)

        assert result == {"zero": 0, "one": 1, "two": 2}


@pytest.mark.parametrize(
    "fields_to_include, expected_result",
    [
        (["zero", "one", "two"], {"zero": 0, "one": 1, "two": 2}),
        (["zero", "two"], {"zero": 0, "two": 2}),
        ([], {}),
    ],
)
def test_row_to_header_dict(fields_to_include, expected_result):
    _dict = {"zero": 0, "one": 1, "two": 2}

    result = filter_dict_fields(_dict=_dict, fields_to_include=fields_to_include)

    assert result == expected_result


@pytest.mark.parametrize(
    "trim_start, trim_end, expected_result",
    [
        (0, 0, [4, 6, 8]),
        (1, 0, [6, 8]),
        (0, 1, [4, 6]),
        (1, 1, [6]),
        (3, 0, []),
        (1, 2, []),
    ],
)
def test_sum_rows(trim_start, trim_end, expected_result):
    list_of_rows = [[1, 2, 3], [3, 4, 5]]

    if trim_start + trim_end > 2:
        with pytest.raises(Exception):
            result = sum_rows(list_of_rows=list_of_rows, trim_start=trim_start, trim_end=trim_end)
    else:
        result = sum_rows(list_of_rows=list_of_rows, trim_start=trim_start, trim_end=trim_end)

        assert result == expected_result
