"""
****************************************
 * @author: Xin Zhang
 * Date: 5/11/21
****************************************
"""

import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import Layout
from xin_util.CreateDIYdictFromDataFrame import CreateDIYdictFromDataFrame


class chain_drop_down:
    def __init__(self, df):
        """
        Note that this only works for categorical columns
        @param df: a dataframe object
        """
        self.df = df
        self.all_widgets = []
        self.intersted_features = []

    @staticmethod
    def loop_dict(logic_dict, key_chain):
        d = logic_dict.copy()
        for k in key_chain:
            d = d[k]
        return d

    def create_dropdown(self, intersted_features, width='30%', height='20px'):
        """

        @param intersted_features: list of column names
        @param width:
        @param height:
        @return:
        """
        df = self.df.dropna(subset=intersted_features)
        df_obj = CreateDIYdictFromDataFrame(df)
        self.intersted_features = intersted_features
        logic_dict = df_obj.DIY_dict(intersted_features, convert_to=lambda x: list(set(x)))

        all_widgets = [
            widgets.Dropdown(
                options=sorted(list(logic_dict.keys())),
                description=intersted_features[0],
                style={'description_width': 'initial'},
                layout=Layout(width=width, height=height)
            )
        ]

        widget_values = []
        for index, feature in enumerate(intersted_features[1:]):
            widget_init = all_widgets[-1].value
            widget_values.append(widget_init)
            new_options = chain_drop_down.loop_dict(logic_dict, widget_values)
            try:
                new_widget = widgets.Dropdown(
                    options=sorted(list(new_options.keys())),
                    description=intersted_features[index + 1],
                    style={'description_width': 'initial'},
                    layout=Layout(width=width, height=height)
                )
            except AttributeError:
                new_widget = widgets.Dropdown(
                    options=sorted(new_options),
                    description=intersted_features[index + 1],
                    style={'description_width': 'initial'},
                    layout=Layout(width=width, height=height)
                )
            all_widgets.append(new_widget)

        self.all_widgets = all_widgets
        name_widget_dict = dict((name, all_widgets[index])
                                for index, name in enumerate(intersted_features))

        @interact(**name_widget_dict)
        def print_widget(**kwargs):
            for index, fea_name in enumerate(intersted_features[:-1]):
                pre_widgets = all_widgets[:index + 1]
                pre_values = [w.value for w in pre_widgets]
                new_options = chain_drop_down.loop_dict(logic_dict, pre_values)
                try:
                    all_widgets[index + 1].options = sorted(list(new_options.keys()))
                except AttributeError:
                    all_widgets[index + 1].options = sorted(new_options)

        return self.all_widgets

    def get_filtered_df(self):
        df_temp = self.df.copy()
        for index, w in enumerate(self.all_widgets):
            feature = self.intersted_features[index]
            df_temp = df_temp[df_temp[feature] == w.value]
        return df_temp


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
