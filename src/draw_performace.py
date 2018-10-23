import matplotlib
matplotlib.use('TkAgg')

from functools import wraps
import  matplotlib.pyplot as plt
import pickle


def track_plot(func):
    plt.ion()
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.called += 1

        loss = func(*args, **kwargs)

        if isinstance(loss, list):
            wrapper.loss += loss
        else:
            wrapper.loss.append(loss)

        plt.clf()
        x = range(len(wrapper.loss))

        plt.plot(x, wrapper.loss, 'g-')
        plt.title('Train Loss')
        plt.ylim(ymax=5000)
        plt.pause(0.00001)
        plt.show(block=False)

        return loss

    wrapper.called = 0
    wrapper.loss = []

    wrapper.__name__ = func.__name__

    return wrapper


def draw(loss, trRSME, valRSME):
    x = range(len(loss))

    plt.plot(x, loss, 'g-')
    plt.title('Train Loss')


    plt.show()





