import pytest

from data.data_collection.utils import row_to_header_dict, filter_dict_fields


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
