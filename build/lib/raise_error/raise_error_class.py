class TestFailed(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


def demo():
    try:
        raise TestFailed('Oops')
    except TestFailed as x:
        print(x)
