def pretty_print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
