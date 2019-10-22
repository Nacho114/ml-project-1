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





def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")