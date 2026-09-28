"""Parser robustness tests (no server). `parse_table` must tokenize the
header with the same quote-aware splitter as the rows: real Freeciv saves emit
quoted column names containing a comma (e.g. "cma_minimal_surplus,1"), which a
bare-comma split would turn into misaligned columns.
"""

from civharness import parse


def test_parse_table_plain():
    raw = '{"id","name","size"\n5,"Rome",7\n6,"Veii",4\n}'
    rows = parse.parse_table(raw)
    assert rows == [
        {"id": "5", "name": "Rome", "size": "7"},
        {"id": "6", "name": "Veii", "size": "4"},
    ]


def test_header_with_quoted_comma_stays_aligned():
    # The 3rd column NAME itself contains a comma inside quotes. A naive
    # split(",") would yield 4 header cells and misalign the row.
    raw = '{"id","name","cma_minimal_surplus,1","size"\n5,"Rome",-3,7\n6,"Veii",0,4\n}'
    rows = parse.parse_table(raw)
    assert len(rows[0]) == 4, rows[0]
    assert "cma_minimal_surplus,1" in rows[0]
    # A misaligned header would shift "size" off its value.
    assert rows[0]["size"] == "7"
    assert rows[0]["cma_minimal_surplus,1"] == "-3"
    assert rows[1]["size"] == "4"


def test_row_with_quoted_comma_value():
    raw = '{"id","name","size"\n6,"a, the great",4\n}'
    rows = parse.parse_table(raw)
    assert rows[0]["name"] == "a, the great"
    assert rows[0]["size"] == "4"


def test_empty_table():
    assert parse.parse_table("{}") == []
    assert parse.parse_table('{"id","name"\n}') == []


def test_unquote():
    assert parse.unquote(' "Rome" ') == "Rome"
    assert parse.unquote("plain") == "plain"
    assert parse.unquote('"') == '"'
