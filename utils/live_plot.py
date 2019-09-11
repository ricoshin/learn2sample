import argparse
import result
from result import ResultDict
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')

parser = argparse.ArgumentParser(description='live plotter for saved result.')
parser.add_argument('--load_dir', type=str)

plot_names = ['SGD', 'RMSprop', 'NAG', 'ADAM']


def live_plotter(x_vec,y1_data,line1,identifier='',pause_time=0.1):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)
        #update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1

def test():
  size = 5
  x_vec = np.linspace(0,1,size+1)[0:-1]
  y_vec = np.random.randn(len(x_vec))
  line1 = []
  import pdb; pdb.set_trace()
  while True:
      rand_val = np.random.randn(1)
      y_vec[-1] = rand_val
      line1 = live_plotter(x_vec,y_vec,line1)
      y_vec = np.append(y_vec[1:],0.0)


def test2():
  # draw the figure so the animations will work
  fig = plt.gcf()
  fig.show()
  fig.canvas.draw()
  size = 100

  while True:
      # compute something
      x_vec = np.linspace(0,100,size+1)[0:-1]
      y_vec = np.random.randn(len(x_vec))* 100
      plt.plot(x_vec, y_vec)

      # update canvas immediately
      plt.xlim([0, 100])
      plt.ylim([0, 100])
      #plt.pause(0.01)  # I ain't needed!!!
      fig.canvas.draw()

class LivePlotter(object):
  def __init__(self, results):
    assert isinstance(results, dict)
    self.results = results


def main():
  args = parser.parse_args()
  results = {}
  for name in plot_names:
    if ResultDict.is_loadable(name, args.load_dir):
      results[name] = ResultDict.load(name, args.load_dir)
    else:
      raise Exception(f"Unalbe to find result name: {name}")

  import pdb; pdb.set_trace()
  size = 5
  x_vec = np.linspace(0,1,size+1)[0:-1]
  y_vec = np.random.randn(len(x_vec))
  line1 = []
  import pdb; pdb.set_trace()
  while True:
      rand_val = np.random.randn(1)
      y_vec[-1] = rand_val
      line1 = live_plotter(x_vec,y_vec,line1)
      y_vec = np.append(y_vec[1:],0.0)

  LivePlotter(results)


if __name__ =="__main__":
  main()
  # test2()
