import matplotlib.pyplot as plt

class Debugger:

    def __init__(self, data_types):
        self.data = {}
        for dt_type in data_types:
            self.data[dt_type] = []

    def add_item(self, data_type, item):
        self.data[data_type].append(item)

    def plot(self, data_type):
        plt.plot(self.data[data_type])
        plt.show()

    def print(self, data_type):
        size = len(self.data[data_type])
        for i, x in enumerate(self.data[data_type]):
            print('step {}/{}'.format(i, size) + ':\t' + data_type + ': ' + str(x))