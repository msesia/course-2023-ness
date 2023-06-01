# Conformal Inference Methods in Deep Learning
*June 4, 2023, Boston (MA).*

This short course provides a hands-on introduction to modern techniques for uncertainty estimation in deep learning with a focus on conformal inference. Assuming only basic prior knowledge of probability, statistics, and machine learning, the course begins with an overview of the key concepts data exchangeability and univariate model-free prediction, which form the foundation of conformal inference. Participants will learn how to leverage these conformal inference ideas to construct reliable and interpretable uncertainty estimates for the predictions of deep neural network models in both multi-class classification and regression problems. The course also covers advanced topics of practical relevance, including techniques for computing conformal inferences that can automatically adapt to possible heteroscedasticity and skewness in the data, methods for obtaining conformal inferences with conditional validity properties to address issues of algorithmic fairness, and techniques for mitigating the over-confidence of deep neural networks. The course includes hands-on coding exercises and real-data demonstrations.


## References

1. "Classification with Valid and Adaptive Coverage", Yaniv Romano, Matteo Sesia, Emmanuel J. Candès. NeurIPS (2020).
2. "Training Uncertainty-Aware Classifiers with Conformalized Deep Learning", Bat-Sheva Einbinder, Yaniv Romano, Matteo Sesia, Yanfei Zhou. NeurIPS (2022).
3. "Conformal inference is (almost) free for neural networks trained with early stopping", Ziyi Liang, Yanfei Zhou, Matteo Sesia. arXiv preprint (2023).
4. "Conformalized Quantile Regression", Yaniv Romano, Evan Patterson, Emmanuel J. Candès. NeurIPS (2019).
5. "A comparison of some conformal quantile regression methods", Matteo Sesia, Emmanuel J. Candès. Stat (2020).
6. "Conformal Prediction using Conditional Histograms", Matteo Sesia, Yaniv Romano. NeurIPS (2021).

## Prerequisites:

 - Basic knowledge of probability and statistics
 - Working knowledge of the Python programming language
 - Basic knowledge of deep learning with PyTorch (optional but recommended)
 - A laptop with Python installed, including key machine learning packages, including Jupyter notebooks, pytorch, scikit-learn, and numpy.


## Software

The computer sessions of this course will utilize Python, PyTorch, and Jupyter notebooks. Students are expected to bring their laptops and have pre-installed Python (version 3.7+), PyTorch, and Jupyter prior to the beginning of the course.
For new (and experienced) users, it is highly recommended to install Anaconda. Anaconda can conveniently install Python, PyTorch, the Jupyter Notebook, and other packages that will be utilized in this course.

Installation instructions: https://docs.jupyter.org/en/latest/install/notebook-classic.html

Anaconda download: https://www.anaconda.com/download 

After you install Python, PyTorch, and Jupyter through Anaconda, read the following start-up guide on how to use Jupyter notebooks: https://realpython.com/jupyter-notebook-introduction/
