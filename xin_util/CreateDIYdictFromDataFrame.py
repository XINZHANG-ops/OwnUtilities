import nltk as nltk
import csv
import re
from functools import reduce
from collections import Counter

path = '/Users/xinzhang/PycharmProjects/State of RFP/'


class Tree(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value


def nested_dict_get(nested_dict, keys):
    return reduce(lambda d, k: d[k], keys, nested_dict)


def nested_dict_set(nested_dict, keys, value):
    nested_dict_get(nested_dict, keys[:-1])[keys[-1]] = value


def nested_dict_update_count(nested_dict, keys):
    if nested_dict_get(nested_dict, keys[:-1]):  # update existing Counter
        nested_dict_get(nested_dict, keys[:-1]).update([keys[-1]])
    else:  # create a new  Counter
        nested_dict_set(nested_dict, keys[:-1], Counter([keys[-1]]))


def dict_type(dictionary, leaves_type=None):
    tree_dict = dict()
    for k, v in dictionary.items():
        if isinstance(v, dict):
            tree_dict[k] = dict_type(v, leaves_type)
        elif isinstance(v, int):
            if leaves_type is None:
                tree_dict = list(dictionary.elements())
            else:
                tree_dict = leaves_type(dictionary.elements())
    return tree_dict


class CreateDIYdictFromDataFrame:
    @staticmethod
    def counter2list(counter):
        return list(counter.elements())

    @staticmethod
    def nest(x):
        return nltk.defaultdict(lambda: x)

    def __init__(self, dataframe):
        InfoDict = dict()
        Headerlist = []
        for header in dataframe.columns:
            Headerlist.append(header)
            InfoDict[header] = list(dataframe[header])
        self.Headerlist = Headerlist
        self.AllInfo = dict(InfoDict)


# valuespatterns is dict or list

    def DIY_dict(
        self, values=None, valuespatterns=None, convert_to=None, only_keep_first_match=True
    ):
        AllInfo = self.AllInfo
        if values is None:
            values = [header for header in self.Headerlist]
        Infolist = [AllInfo[value] for value in values]
        Range = len(Infolist[0])
        DIYdict = Tree()
        for n in range(Range):
            Keys = []
            for i, header in enumerate(values):
                try:
                    if type(valuespatterns) is dict:
                        pattern = valuespatterns[header]
                    else:
                        pattern = valuespatterns[i]
                    Find = re.findall(pattern, Infolist[i][n])
                    if only_keep_first_match:
                        Find.append('NOT FIND')
                        Keys.append(Find[0])
                    else:
                        if Find:
                            Keys.append(tuple(Find))
                        else:
                            Find.append('NOT FIND')
                            Keys.append(tuple(Find))
                except:
                    Keys.append(Infolist[i][n])
            nested_dict_update_count(DIYdict, Keys)
        if convert_to is None:
            return DIYdict
        else:
            DIYdict = dict_type(DIYdict, leaves_type=convert_to)
            return DIYdict
