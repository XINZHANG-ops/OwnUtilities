from raise_error.raise_error_class import TestFailed
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class Categorical_Scatter:
    def __init__(self, X, y=None):
        """

        @param X: array, rows are samples, columns are features, 3 features at most
        @param y: array, indicates the categories for each data point
        """
        self.X = X
        if y is None:
            self.y = np.ones(X.shape[0])
        else:
            self.y = y

    @staticmethod
    def plot3d_plotly(X, y, marker_size=7, width=500, height=500):
        # x has 3 columns, y indicates the labels
        import random
        random.seed(0)
        r = lambda: random.randint(0, 255)
        color = '#%02X%02X%02X' % (r(), r(), r())
        unique_labels = sorted(list(set(y)))
        y = np.array(y)
        color0 = list(np.random.choice(range(256), size=3))
        df_0 = {
            'x': X[y == unique_labels[0], 0],
            'y': X[y == unique_labels[0], 1],
            'z': X[y == unique_labels[0], 2]
        }
        df_0 = pd.DataFrame(df_0)
        fig = go.Figure(
            data=go.Scatter3d(
                x=df_0['x'],
                y=df_0['y'],
                z=df_0['z'],
                mode='markers',
                name=f"type {unique_labels[0]}",
                marker=dict(
                    size=marker_size,
                    color=color,
                    colorscale='Viridis',
                )
            )
        )
        for label in unique_labels[1:]:
            r = lambda: random.randint(0, 255)
            color = '#%02X%02X%02X' % (r(), r(), r())
            df = {'x': X[y == label, 0], 'y': X[y == label, 1], 'z': X[y == label, 2]}
            df = pd.DataFrame(df)
            fig.add_trace(
                go.Scatter3d(
                    x=df['x'],
                    y=df['y'],
                    z=df['z'],
                    mode='markers',
                    name=f"type {label}",
                    marker=dict(
                        size=marker_size,
                        color=color,
                        colorscale='Viridis',
                    )
                )
            )
        fig.update_layout(
            width=width,
            height=height,
            autosize=False,
            scene=dict(
                camera=dict(up=dict(x=0, y=0, z=1), eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual'
            ),
        )
        fig.show()

    @staticmethod
    def plot2d_plotly(X, y, marker_size=7, width=500, height=500):
        import random
        random.seed(0)
        r = lambda: random.randint(0, 255)
        unique_labels = sorted(list(set(y)))
        y = np.array(y)
        fig = go.Figure()
        for label in unique_labels:
            color = '#%02X%02X%02X' % (r(), r(), r())
            df = {'x': X[y == label, 0], 'y': X[y == label, 1]}
            fig.add_trace(
                go.Scatter(
                    x=df['x'],
                    y=df['y'],
                    mode='markers',
                    name=f"type {label}",
                    marker=dict(
                        size=marker_size,
                        color=color,
                        colorscale='Viridis',
                    )
                )
            )
        fig.update_layout(
            width=width,
            height=height,
            autosize=False,
            scene=dict(
                camera=dict(up=dict(x=0, y=0, z=1), eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual'
            ),
        )
        fig.show()

    def plot(self, marker_size=7, width=500, height=500):
        n_sample, n_dim = self.X.shape
        if n_dim == 2:
            Categorical_Scatter.plot2d_plotly(self.X, self.y, marker_size=marker_size, width=width, height=height)
        elif n_dim == 3:
            Categorical_Scatter.plot3d_plotly(self.X, self.y, marker_size=marker_size, width=width, height=height)
        else:
            raise TestFailed('X must be 2 or 3 dimensions')


def demo():
    def generate_spheres(dim=3, radiuses=(1, 2, 3), n_sample_each=300):
        """
        generate data on spheres (in respective dimension)

        @param dim: data space dimension
        @param radiuses: tuple, how many different spheres and their radius
        @param n_sample_each: how many samples for each category
        @return:
        """
        def sample_spherical(radius, npoints, ndim):
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0)
            return vec * radius

        X_origin = np.array([])
        y = np.array([])
        for index, radius in enumerate(radiuses):
            if index == 0:
                X_origin = sample_spherical(radius, n_sample_each, dim).transpose()
            else:
                X_origin = np.concatenate(
                    (X_origin, sample_spherical(radius, n_sample_each, dim).transpose())
                )
            y = np.concatenate((y, np.ones(n_sample_each) * radius))
        return X_origin, y

    # generate 3d data and plot
    X_origin, y = generate_spheres(dim=3, radiuses=(1, 2, 3, 4), n_sample_each=300)
    cs = Categorical_Scatter(X_origin, y)
    cs.plot()

    # use kernel pca project to 2d
    from sklearn import decomposition
    kernelpca = decomposition.KernelPCA(n_components=2, kernel='rbf')
    kernelpca.fit(X_origin)
    X = kernelpca.transform(X_origin)
    cs = Categorical_Scatter(X, y)
    cs.plot()
