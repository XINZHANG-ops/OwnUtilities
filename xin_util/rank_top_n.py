"""
****************************************
 * @author: Xin Zhang
 * Date: 5/19/21
****************************************
"""

def find_top_n_faster(vlist, top_n, method="min"):
    print(f'finding top {top_n} values')
    top_n_value_dict = {0: vlist[0]}
    top_n_index_dict = {0: 0}
    current_extreme_value = vlist[0]
    current_extreme_key = 0
    for i in range(top_n - 1):
        new_value = vlist[i + 1]
        new_index = i + 1
        top_n_value_dict[new_index] = new_value
        top_n_index_dict[new_index] = new_index
        top_n_inverse_value = {v: k for k, v in top_n_value_dict.items()}
        if new_value < current_extreme_value and method == 'min':
            current_extreme_value = max(top_n_inverse_value.keys())
            current_extreme_key = top_n_inverse_value[current_extreme_value]
        elif new_value > current_extreme_value and method == 'max':
            current_extreme_value = min(top_n_inverse_value.keys())
            current_extreme_key = top_n_inverse_value[current_extreme_value]
        else:
            continue

    for idx, value in enumerate(vlist[top_n:]):
        list_index = idx + top_n
        if value < current_extreme_value and method == 'min':
            top_n_value_dict[current_extreme_key] = value
            top_n_index_dict[current_extreme_key] = list_index
            top_n_inverse_value = {v: k for k, v in top_n_value_dict.items()}
            current_extreme_value = max(top_n_inverse_value.keys())
            current_extreme_key = top_n_inverse_value[current_extreme_value]
        elif value > current_extreme_value and method == 'max':
            top_n_value_dict[current_extreme_key] = value
            top_n_index_dict[current_extreme_key] = list_index
            top_n_inverse_value = {v: k for k, v in top_n_value_dict.items()}
            current_extreme_value = min(top_n_inverse_value.keys())
            current_extreme_key = top_n_inverse_value[current_extreme_value]
        else:
            continue

    return list(top_n_value_dict.values()), list(top_n_index_dict.values())


def find_top_n(vlist, top_n, method="min"):
    print(f'finding top {top_n} values')
    top_n_l = vlist[:top_n]
    top_n_index = list(range(top_n))
    if method == 'max':
        for idx, v in enumerate(vlist[top_n:]):
            min_in_l = min(top_n_l)
            min_idx = top_n_l.index(min_in_l)
            if v > min_in_l:
                top_n_l[min_idx] = v
                top_n_index[min_idx] = idx + top_n

    elif method == 'min':
        for idx, v in enumerate(vlist[top_n:]):
            max_in_l = max(top_n_l)
            max_idx = top_n_l.index(max_in_l)
            if v < max_in_l:
                top_n_l[max_idx] = v
                top_n_index[max_idx] = idx + top_n
    return top_n_l, top_n_index


def demo():
    """
    compare 2 algo to pick top n from
    @return:
    """
    import numpy as np
    import time
    a = list(np.random.uniform(low=-1.0, high=1.0, size=200000))
    top_n = 1000
    method = 'min'

    start = time.time()
    x1, y1 = find_top_n_faster(a, top_n, method)
    print(x1, y1)
    print(time.time() - start)

    start = time.time()
    x2, y2 = find_top_n(a, top_n, method)
    print(x2, y2)
    print(time.time() - start)

    print('verify------')
    print([a[i] for i in y1])
    print([a[i] for i in y2])
