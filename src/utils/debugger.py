import matplotlib.pyplot as plt

class Debugger:
    '''Debugger class to have a callable debuger'''

    def __init__(self, data_types):
        self.data = {}
        for dt_type in data_types:
            self.data[dt_type] = []

    def add_item(self, data_type, item):
        self.data[data_type].append(item)

    def plot(self, data_type):
        plt.plot(self.data[data_type])
        plt.show()

    def clear(self):
        for dt_type in self.data:
            self.data[dt_type] = []

    def print(self, data_type, last_n=0):
        size = len(self.data[data_type])

        if last_n == 0:
            last_n = size

        for i, x in enumerate(self.data[data_type][-last_n:]):
            print('step {}/{}'.format(size - last_n + i + 1, size) + ':\t' + data_type + ': ' + str(x))