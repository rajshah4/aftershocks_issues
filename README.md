# aftershocks_issues
## Issues with Deep Learning of Aftershocks by DeVries


This repo focuses on issues noted by me on by DeVries, et al., *[Deep learning of aftershock patterns following large Earthquakes](https://www.nature.com/articles/s41586-018-0438-y )* or via [sci-hub](https://sci-hub.se/http://www.nature.com/articles/s41586-018-0438-y).  This article has been widely used as a motivation for using deep learning, e.g., [Tensorflow 2.0 release notes](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8).  

I raised concerns about target leakage and the suitability of the data science approach to both the author and Nature.  Nature reviewed my concerns and decided not to act.  You can view the detail of this communication in the [correspondence folder](https://github.com/rajshah4/aftershocks_issues/tree/master/correspondence).  

The repo here demonstrates the issues I noted.  The repo is a [clone of the original analysis](https://github.com/phoebemrdevries/Learning-aftershock-location-patterns).  To understand the issues, work through the notebook, [Exploratory Analysis](https://github.com/rajshah4/aftershocks_issues/blob/master/Exploratory%20Analysis.ipynb).  To run these, you will need the data, which is available at on [google drive](https://drive.google.com/drive/folders/1lAHfdjFd-Uv3wJcA0Tk2ViDIZeFt0mCA?usp=sharing).  You may also want to see how the original test/train splits were conducted at [DeVries processing repo](https://github.com/phoebemrdevries/Process-Srcmod-Files).

To run the notebook, the code is using Python 3 and you must first download the data and put it in an adjoining folder to the repo.

The notebook has four sections:

1. Replicating the results in the paper
2. Replicating the results in the paper, but showing the results on both test and train. Puzzingly, the scores for the test set are higher than the train set.
3. Replicating similar results using only 1500 rows of data with 2 epochs (The original paper used 4.7 million rows of data).
4. One source of potential leakage in how test/train is constructed


I want to thank Lukas Innig and Shubham Cheema for their assistance, as well as all the great data scientists at DataRobot which supported me through this process.

Recently, I found papers by Arnaud Mignan and Marco Broccardo that identify issues in the aftershocks paper, see:
*One neuron is more informative than a deep neural network for aftershock pattern forecasting*, [Arxiv](https://arxiv.org/abs/1904.01983) and *A Deeper Look into ‘Deep Learning of Aftershock Patterns Following Large Earthquakes’: Illustrating First Principles in Neural Network Physical Interpretability* - [Springer](https://link.springer.com/chapter/10.1007/978-3-030-20521-8_1)


