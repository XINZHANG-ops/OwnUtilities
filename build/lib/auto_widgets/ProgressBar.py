"""
****************************************
 * @author: Xin Zhang
 * Date: 5/20/21
****************************************
"""
from ipywidgets import IntProgress, Layout
from IPython.display import display
import time


class progress_bar:
    def __init__(self, min_value, max_value, width='30%', height='30px'):
        self.progressbar = IntProgress(
            min=min_value,
            max=max_value,
            style={'description_width': 'initial'},
            layout=Layout(width=width, height=height)
        )
        display(self.progressbar)


def demo():
    max_count = 100
    f = progress_bar(0, max_count)
    count = 0
    while count <= max_count:
        f.progressbar.value += 1
        f.progressbar.description = '%.2f' % round(f.progressbar.value / max_count, 2) + '%'
        time.sleep(.1)
        count += 1
