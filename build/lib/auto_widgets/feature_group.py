import numpy as np
import ipywidgets as widgets
from ipywidgets import Layout
from ipywidgets import interact
from plotly import graph_objects as go


class TypeNoMatch(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


class create_feature_group:
    def __init__(self, group_definitions, target_df):
        group_definitions_temp = group_definitions.copy()
        categorical_features = []
        numerical_features = []
        for gd in group_definitions_temp:
            if gd['type'].lower() == 'categorical':
                gd['features'] = gd['features'].lower()
                gd['values'] = [v.lower() for v in gd['values']]
                categorical_features.append(gd['features'])
            elif gd['type'].lower() == 'numerical':
                gd['features'] = [f.lower() for f in gd['features']]
                numerical_features.extend(gd['features'])
            else:
                raise TypeNoMatch('type needs to be categorical or numerical')

        self.group_definitions = group_definitions_temp
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        df = target_df.copy()
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(subset=self.categorical_features)
        df[self.numerical_features] = df[self.numerical_features].fillna(0)
        df = df.applymap(lambda s: s.lower() if type(s) == str else s)
        self.target_df = target_df
        self.df = df

    def create_funnel(self, width=800, height=1000, autosize=False):
        df = self.df.copy()
        data_size_remain = [df.shape[0]]
        y_names = ['origin_df']

        for index, gdf in enumerate(self.group_definitions):
            current_method = gdf['type']
            name = gdf['name']
            if current_method == 'categorical':
                features = gdf['features']
                values = set(gdf['values'])
                df = df[df[features].isin(values)]
            else:
                features = gdf['features']
                values = gdf['values']
                vmin = values[0]
                vmax = values[1]
                df['combined'] = np.zeros(df.shape[0])
                for f in features:
                    df['combined'] += df[f]
                df = df[(vmin <= df['combined']) & (vmax >= df['combined'])]
            y_names.append(name)
            data_size_remain.append(df.shape[0])

        fig = go.Figure(go.Funnel(y=y_names, x=data_size_remain))
        fig.update_layout(
            autosize=autosize,
            width=width,
            height=height,
            margin=dict(l=50, r=50, b=50, t=50, pad=4),
            paper_bgcolor="LightSteelBlue"
        )
        fig.show()

    def get_df(self, origin=True):
        df = self.df.copy()
        for index, gdf in enumerate(self.group_definitions):
            current_method = gdf['type']
            if current_method == 'categorical':
                features = gdf['features']
                values = set(gdf['values'])
                df = df[df[features].isin(values)]
            else:
                features = gdf['features']
                values = gdf['values']
                vmin = values[0]
                vmax = values[1]
                df['combined'] = np.zeros(df.shape[0])
                for f in features:
                    df['combined'] += df[f]
                df = df[(vmin <= df['combined']) & (vmax >= df['combined'])]
        try:
            df.drop(columns=['combined'], inplace=True)
        except KeyError:
            pass
        if origin:
            return self.target_df.iloc[df.index]
        return df

    def create_drop_down_with_funnel(
        self,
        width='30%',
        height='20px',
        plot_starting_from=0,
        funnel_width=800,
        funnel_hight=1000,
        autosize=False
    ):
        df = self.df.copy()
        all_widgets = []
        all_feature_name = []
        for index, gdf in enumerate(self.group_definitions):
            current_method = gdf['type']
            features = gdf['features']
            values = gdf['values']
            name = gdf['name']
            if current_method == 'categorical':
                values_dict = dict((v, name) for v in values)
                df.replace({features: values_dict}, inplace=True)
                w = widgets.Dropdown(
                    options=sorted(df[features].unique().tolist()),
                    value=name,
                    description=f'{features}:',
                    style={'description_width': 'initial'},
                    layout=Layout(width=width, height=height)
                )
                all_feature_name.append(features)
            else:
                vmin = values[0]
                vmax = values[1]
                step = gdf['step']
                df[name] = np.zeros(df.shape[0])
                for f in features:
                    df[name] += df[f]
                w = widgets.FloatRangeSlider(
                    value=[vmin, vmax],
                    min=df[name].min(),
                    max=df[name].max(),
                    step=step,
                    description=f'{name}:',
                    style={'description_width': 'initial'},
                    layout=Layout(width=width, height=height)
                )
                all_feature_name.append(name)
            all_widgets.append(w)

        def plot_funnul_all(**kwargs):
            all_names = all_feature_name.copy()
            data_size_remain = []
            clean_df_temp = df.copy()
            for index, w in enumerate(all_widgets):
                current_col = all_names[index]
                current_method = self.group_definitions[index]['type']
                if current_method == 'categorical':
                    clean_df_temp = clean_df_temp[clean_df_temp[current_col] == w.value]
                else:
                    vmin = w.value[0]
                    vmax = w.value[1]
                    clean_df_temp = clean_df_temp[(vmin <= clean_df_temp[current_col])
                                                  & (vmax >= clean_df_temp[current_col])]
                data_size_remain.append(clean_df_temp.shape[0])

            self.dynamic_df = clean_df_temp

            all_names = ['original'] + all_names
            data_size_remain = [df.shape[0]] + data_size_remain

            if isinstance(plot_starting_from, str):
                starting_idx = all_names.index(plot_starting_from)
            elif isinstance(plot_starting_from, int):
                starting_idx = plot_starting_from
            else:
                starting_idx = 0
            data_size_remain = data_size_remain[starting_idx:]
            all_names = all_names[starting_idx:]

            fig = go.Figure(go.Funnel(y=all_names, x=data_size_remain))
            fig.update_layout(
                autosize=autosize,
                width=funnel_width,
                height=funnel_hight,
                margin=dict(l=10, r=10, b=10, t=10, pad=1),
                paper_bgcolor="LightSteelBlue"
            )
            fig.show()

        widget_dict = dict((all_feature_name[index], w) for index, w in enumerate(all_widgets))
        interact(plot_funnul_all, **widget_dict)
        return all_widgets


def demo():
    import pandas as pd
    import numpy as np
    from random import choices, uniform, randrange

    # define a dataframe to play with
    data_size = 10000
    df = pd.DataFrame(np.zeros((data_size, 6)))
    name_dict = dict((i, f'col{i}') for i in range(4))
    df = df.rename(columns=name_dict)
    col0 = choices(['ford', 'audi', 'bmw', 'mercedes'], k=data_size)
    col1 = choices(['truck', 'coupe', 'sedan', 'sport'], k=data_size)
    col2 = choices(range(100, 10000), k=data_size)
    col3 = np.random.uniform(low=0, high=1, size=data_size)
    col4 = np.random.uniform(low=0, high=1, size=data_size)
    col5 = np.random.uniform(low=0, high=1, size=data_size)

    # we can think of col3,4,5 are some numerical features of car
    df['col0'] = col0
    df['col1'] = col1
    df['col2'] = col2
    df['col3'] = col3
    df['col4'] = col4
    df['col5'] = col5

    # d1 we define a wanted feature from col0 for ford and bmw
    d1 = {
        'type': 'categorical',
        'features': 'col0',
        'values': ['ford', 'BMW'],
        'name': 'brand',
        'step': None
    }

    d2 = {
        'type': 'categorical',
        'features': 'col1',
        'values': ['coupe', 'sedan'],
        'name': 'common type',
        'step': None
    }

    # d3 we can think of we can combine two numerical features as a new feature to filter
    # where step is the step for wiget step
    # values is the range(low, high) we want the feature to be in
    d3 = {
        'type': 'numerical',
        'features': ['col3', 'col4'],
        'values': [0.3, 1],
        'name': 'col3 add col4',
        'step': 0.1
    }

    d4 = {
        'type': 'numerical',
        'features': ['col5'],
        'values': [0.5, 1],
        'name': 'col5 alone',
        'step': 0.01
    }

    group_definitions = [d1, d2, d3, d4]

    cfg = create_feature_group(group_definitions, df)
    print('the categorical features:')
    print(cfg.categorical_features)
    print()
    print('the numerical features:')
    print(cfg.numerical_features)
    print()
    print('the group definitions:')
    print(cfg.group_definitions)
    all_widgets = cfg.create_drop_down_with_funnel(
        plot_starting_from=0, funnel_width=500, funnel_hight=300
    )
    print('filter dataframe by selection')
    cfg.dynamic_df
