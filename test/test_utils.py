from torchtree.core.utils import string_to_list_index


def test_string_to_list_index():
    a = [0, 1, 2, 3, 4, 5]
    assert a[string_to_list_index('2')] == 2
    assert a[string_to_list_index(':2')] == a[:2]
    assert a[string_to_list_index('2:')] == a[2:]
    assert a[string_to_list_index('1:3')] == a[1:3]
    assert a[string_to_list_index(':5:2')] == a[:5:2]
    assert a[string_to_list_index('1::2')] == a[1::2]
    assert a[string_to_list_index('1:2:')] == a[1:2:]
    assert a[string_to_list_index('1:5:2')] == a[1:5:2]
    assert a[string_to_list_index('::-1')] == a[::-1]
