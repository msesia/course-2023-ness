# Conformal Inference Methods in Deep Learning
*June 4, 2023, Boston (MA).*

This short course provides a hands-on introduction to modern conformal inference techniques for uncertainty estimation in deep learning.
Assuming only basic prior knowledge of probability, statistics, and machine learning, the course begins with an overview of the key concepts data exchangeability and univariate model-free prediction, which constitute the foundation of conformal inference. Participants will learn how to leverage conformal inference ideas to construct reliable and interpretable uncertainty estimates for the predictions of deep neural network models in both regression and classification problems.
The course also covers advanced topics of practical relevance, including techniques for computing conformal inferences that can automatically adapt to possible heteroscedasticity in the data, methods for obtaining conformal inferences with conditional validity properties to address issues of algorithmic fairness, and cross-validation approaches to make an efficient use of the available data. The course includes hands-on coding exercises and real-data demonstrations.



## References

1. "Distribution-Free Predictive Inference For Regression", J. Lei, M. G'Sell, A. Rinaldo, R. Tibshirani, L. Wasserman. JASA (2017)
2. "Conformalized Quantile Regression", Y. Romano, E. Patterson, E. Candès. NeurIPS (2019).
3. "A comparison of some conformal quantile regression methods", M. Sesia, E. Candès. Stat (2020).
4. "Classification with Valid and Adaptive Coverage", Y. Romano, M. Sesia, E. Candès. NeurIPS (2020).
5. "With malice toward none: Assessing uncertainty via equalized coverage", Y. Romano, R.F. Barber, C. Sabatti, E. Candès. Harvard Data Science Review (2020)
6. "Predictive inference with the jackknife+", R.F. Barber, E. Candès, A. Ramdas, R. Tibshirani. Ann. Statist. (2021)


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
