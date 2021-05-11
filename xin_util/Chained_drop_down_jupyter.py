"""
****************************************
 * @author: Xin Zhang
 * Date: 5/11/21
****************************************
"""

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
from .CreateDIYdictFromDataFrame import CreateDIYdictFromDataFrame


class chain_drop_down:
    def __init__(self, df):
        """
        Note that this only works for categorical columns
        @param df: a dataframe object
        """
        self.df = df
        self.all_widgets = []

    def create_dropdown(self, intersted_features):
        """

        @param intersted_features: a list of str, col names of df(the order matters)
        @return:
        """
        df = self.df.dropna(subset=intersted_features)
        df_obj = CreateDIYdictFromDataFrame(df)
        all_dicts = []
        for i in range(len(intersted_features) - 1):
            all_dicts.append(
                df_obj.DIY_dict([intersted_features[i], intersted_features[i + 1]],
                                convert_to=lambda x: list(set(x)))
            )
        all_widgets = [
            widgets.Dropdown(
                options=sorted(list(all_dicts[0].keys())), description=intersted_features[0]
            )
        ]
        for index, feature in enumerate(intersted_features[1:]):
            widget_init = all_widgets[-1].value
            new_widget = widgets.Dropdown(
                options=sorted(list(all_dicts[index][widget_init])),
                description=intersted_features[index + 1]
            )
            all_widgets.append(new_widget)

        self.all_widgets = all_widgets
        name_widget_dict = dict((name, all_widgets[index])
                                for index, name in enumerate(intersted_features))

        @interact(**name_widget_dict)
        def print_city(**kwargs):
            for index, fea_name in enumerate(intersted_features[:-1]):
                all_widgets[index + 1].options = all_dicts[index][kwargs.get(fea_name)]

        return self.all_widgets


def demo():
    import pandas as pd
    df = pd.DataFrame([['US', 'New york', 'd1'], ['US', 'New york', 'd2'], ['US', 'Chicago', 'd3'],
                       ['US', 'Chicago', 'd4'], ['ENG', 'London', 'd5'], ['ENG', 'London', 'd6'],
                       ['ENG', 'Manchester', 'd7'], ['ENG', 'Manchester', 'd8']],
                      columns=['county', 'city', 'district'])
    cdd = chain_drop_down(df)
    all_widgets = cdd.create_dropdown(['county', 'city', 'district'])

    for w in all_widgets:
        print(w.value)
