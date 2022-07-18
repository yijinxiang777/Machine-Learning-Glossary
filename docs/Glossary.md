---
layout: default
title: Configuration
nav_order: 2
---

# Machine Learning and Artificial Intelligence: Glossary of Terms]
{: .no_toc }

Just the Docs has some specific configuration parameters that can be defined in your Jekyll site's \_config.yml file.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---


As many of the examples above demonstrate, the successful implementation of any machine learning or artificial intelligence algorithms depends on an awareness of how these methods perform in different settings, and on the options available to researchers for tailoring these methods to the problems and data at hand. To this end, we provide here a contextualized glossary of terms to facilitate understanding of some of the details required to appropriately use ML/AI methods in epidemiologic research. Our glossary is contextualized insofar as we attempt to directly link concepts in ML/AI with epidemiologic concepts and terms. As we will show, relying on many of the core concepts and practices in epidemiology can do much to improve the quality and interpretability of ML/AI implementations. 

  
#### supervised learning  

Regressing an outcome (referred to as labels) regressed against covariates (or features).
The goal is to use data to produce a model that takes features as input and outputs information about label.  

#### unsupervised learning  

An analysis of unlabeled (i.e., no outcome) data.
A collection of unlabeled variables is used to create a model that solves “practical problems” such as cluster detection, dimension reduction, or outlier detection.  

#### semi-supervised learning  

It is the same as supervised learning, except that there are more feature vectors (covariates) than there are labeled outputs (outcomes). The rationale is to improve performance of supervised learning model using more feature vectors. The goal is typically to develop an algorithm that predicts outcomes or classifies observations with both missing and observed outcome values.
We can also call it regression with missing data - to train models on complete data, and use these models to predict outcomes for observations without outcomes.  

####  reinforcement learning   

Machine lives in an environment and is capable of perceiving state. Machine executes actions that yield different rewards and move machine into another state. Goal of reinforcement learning is for machines to learn a “policy” - a function that takes feature vector as input (state) and outputs optimal action (i.e., maximizing expected average reward).

#### intelligible/unintelligible 
There is a tradeoff between accuracy and the intelligible/unintelligible nature of machine learning algorithms. More specifically, algorithms that are unintelligible (‘black box’) means that terms cannot be readily identified and manipulated. Intelligible models, on the other hand, do have terms that can be identified and modified (generalized additive models), but may sacrifice accuracy.

#### Label and class  

[ain: make distinctions between risk prediction and classification, regression and classification, etc]
Label is analogous to outcome. The label yi  can be a element belonging to a finite set of categories 1,2,⋯,C, or a real number, or a more complex structure, like a vector, or a matrix. 
A set of values for a label with finite categories is called classes. For example, a set with two values can denote the class of any dichotomous outcome, such as {Yes, No}, {1, 0}.
Labeled example is a dataset containing both outcome (label) and covariates (features), which usually served as an input for supervised learning. It is denoted as  {(xi,yi)}i=1N.
Unabled example is a dataset with only features (covariates) but no label (outcome), which usually served as an input for unsupervised learning. It is usually denoted as {(xi)}i=1N.  

#### Classification and  Regression   

Classification is a problem of automatically assigning a label to an unlabeled example. A classification model will be learned from a set of labeled examples, which can take unlabeled examples as input and either directly assign a label or a number that can be used to deduce the label. Probability is one example of the number.  There are two types of classification: binary (binominal) classification and multiclass (multinomial) classification. Logistics regression and SVM are two common examples of classification algorithms.
Regression is a problem of predicting a continuous label to an unlabeled example, which can take unlabeled examples as input and directly assign a continuous label. Linear regression is an example of regression algorithms.  


#### Parameters and hyperparameters  

Parameters are part of the model learned from the dataset, which usually doesn’t need to be set by users. For example, the slope/slopes and intercept for regression.

Hyperparameters are the innate properties of each algorithm, which needs to be set in advance or tuned by users. For example, number of knots and polynomial degrees are two hyperparameters for Spline regression. Another common example of hyperparameter in ML would be the hyperparameter C for SVM,  determining the tradeoff between increasing the size of the decision boundary and ensuring the accuracy of assigning labels to each example.
The experimental process of finding the best combination of values for hyperparameters is called tuning. Grid search is the most simple hyperparameter tuning technique.

#### Margin 2

The distance between the closest examples of two classes, as defined by the decision boundary. A large margin contributes to a better generalization - the performance of the model when applied to classify external dataset in the future.

#### Kernels 2  

(Yijin’s comment: I am quite confused about this term. I think it is used when the linear decision boundary can’t separate the examples well for SVM in the book, but I found it might also be used in other algorithms when googling.) [ain: not sure how useful this will be, but there is a core definition that i think we can try to articulate]

#### Dimension and Curse of Dimensionality 2 
The position of an attribute in the vector. For example, index j of xj denotes the jth attribute in the vector x.

#### Curse of dimensionality [ashley send]
The “curse of dimensionality”, a term first coined by Bellman,  describes the difficulties in finding computational solutions to a series of analytic equations when the number variables in the system is large.\cite{Bellman1957}$^{(ix)}$ \cite{Wasserman2006} Many different types of analytic equations are implicated, including regression models that are commonly used in epidemiology.
A first concept to understand is the “dimension” of a model. Consider a simple parametric regression model:
E(Y | C) = 0 + 1C1+ ... + 10C10

In this instance, when the vector of covariates C is a set of binary variables, the dimension of this simple parametric model is 10. As additional variable transformations are included in the model, such as two-way interaction terms, the dimension of the model increases. For example, including all two-way interaction terms would lead to a model dimension number of:
 10 + (10!2!(10-2)!) = 55. 
In the setting where all variables in C are binary, one can fully saturate the model by including all possible interactions, including two-way, three-way, … up to 10-way interactions. In this case, the model dimension for 10 binary variables becomes:
210 - 1 = 1,023
In this example, adding more interaction terms to a model with 10 binary covariates would, in principle, increase the model’s flexibility, thus leading to lower bias. However, for a fixed sample size, the addition of such a large number of terms in the model will increase the variance such that the overall mean squared error of the model is sub-optimal. 
Fundamentally, the curse of dimensionality is the result of this sub-optimal bias-variance tradeoff. Specifically, one may opt to keep the mean squared error for a given regression model smaller than some arbitrary threshold . It’s straightforward to show that the sample size needed to keep MSE low increase exponentially as the number of covariates + transformations (e.g., interaction terms) increases linearly. This usually results in a need for sample sizes much larger than physically (or materially) possible. This exponential increase in the sample size requirement to keep MSE arbitrarily low is the curse of dimensionality. 
In a practical sense, one consequence of the curse of dimensionality is that we often cannot simply fit fully saturated models to a dataset of interest.  In this case, we may encounter model fitting problems that result in errors or warnings from our statistical software programs (e.g., complete or quasi-complete separation of points for a logistic regression model). However, even if our software algorithms run without errors or warnings, the may end up fitting a model with so many terms that the model variance is so large that it is uninformative. This is the consequence of the CoD. 
How does this apply to ML? How does parametric regression avoid problems from CoD?
This example is specific to the case where C includes only binary variables. The problems that result from the curse of dimensionality are only exacerbated with more complex variables (ordinal, nominal, continuous).

Machine learning algorithms are maximizing flexibility and minimizing variance; when you fit a random forest, the reason you don’t create a branch at every possible point in the data, you will end up with an enormous variance. ML algorithms try to optimize the bias-variance trade off.

#### model based learning and instance-based learning   

Model-based learning algorithms train parameters for specific models by learning training data.
Instance-based learning algorithms assign a label seen most frequently in the close neighborhood of the input example to an input.

#### shallow learning and deep learning   

Shallow learning is a method learning the parameters of the model directly from the features of data. Shallow learning algorithms require careful engineering and considerable domain expertise to transform the raw data (such as the pixel values of an image) into a suitable internal representation or feature vector from which the algorithm, could detect or classify patterns in the input. Their ability to process natural data in their raw form is very limited. 

Deep learning is a method allowing computational models that are composed of multiple processing layers to learn representations of raw data with multiple levels of abstraction.

The key aspect of deep learning is that layers of features are not pre-disgned manually, but learned from data using a general-propose learning procedure. Deep learning has shown to be powerful at revealing intricate structures in high-dimensinal data. In addition to prevailing in image recognition and speech recognition, it has been widely used in predicting the activity of potential drug molecules, reconstructing brain circuits, and predicting the effects of mutations in non-coding DNA on gene expression and disease.

#### loss function and cost function 

Loss function is generally considered as the error between the prediction and the true output, varying with every training example. The average loss function over the entire training dataset is known as cost function, which is also called empirical risk. For instance, we want to minimize the expression (fw,b(Xi)-yi)2 in linear regression fw,b(X)=WX+b for every training example i, where fw,b(Xi) is the predicted value and yi is the true value. We call the 1Ni=1N(fw,b(Xi)-yi)2 as the corresponding cost function. 1Ni=1N(fw,b(Xi)-yi)2is the formula for mean squared error (MSE) calculation. 
The process of minizing the expectation of the loss function is called empirical risk minimization.

#### Optimization algorithm [gradient descent and stochastic gradient descent]  

Two most frequently used optimization algorithms found in cases where the optimization criterion (i.e., a cost function) is differentiable.
Gradient descent is an interactive algorithm and aims to find a local minimum of a function, by starting at some random point and taking steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. (Yijin’s Comment: I still find this expression a little bit confusion, not sure whether we should provide an example here)

#### overfitting and underfitting  

[ain: We should consider adding smoothing, and bias variance tradeoff]
Overfitting refers to a model that learns noise and details of data entities in the dataset and performs well on the training data while not on the testing data.  The sample size rule of thumb for linear regression to make sure at least 20 cases per independent variable is an example for avoiding overfitting. 
Underfitting refers to a model that can neither predict the training data well nor generalize to testing data. Increasing model complexity, including including more covariates and combinations of covariates (i.e., interaction terms) is an example for reducing underfitting. 
Smoothing algorithm is one of the strategy for improving model fit. It filter out noise by summarizing a local or global domain of outcome, resulting in an estimation of the underlying data.
Bias is an error from inappropriate assumptions in the learning algorithm (i.e., insufficient number of features). Algorithms with high bias can misrepresent the relations between exposures and outcome (i.e., underfitting). Variance is an error from sensitivity to small fluctuations (e.g., noises) in the training dataset. Algorithms with high variance can have poor generalizabilities on testing data (i.e., overfitting). 
Bias–variance tradeoff is a property of a model that variance of the estimates can be reduced by increasing the bias.

#### Regularization

Regularization methods force the learning algorithm to build a less complex model,  by adding a penalizing term whose value is higher when the model is more complex. In practice, that often leads to slightly higher bias but significantly reduces the variance.
Firth logistics regression is an example of regularization. Iteratively updated weights were added to the response value yi and 1-yi, gurateeing finit estimates and eliminating the problem of separation, to reduce the bias of maximum likelihood estimates in generalized linear models/

#### confusion matrix 

A table comparing predicted values generated by a classification model and true values.

|               | Yes(Predicted)      | No(Predicted)       | 
|---------------|---------------------|---------------------|
| **Yes(True)** | True Positive (TP)  | False Negative (FN) |
| **No(True)**  | False Positive (FP) | True Negative (TN)  |
  

#### precision and recall

We also call precision as positive predictive value (PPV) and call recall as sensitivity.
$$precision\overset{def}{=}\frac{true \ positive}{true \ positive + false \ postive}$$  
$$recall\overset{def}{=}\frac{true \ positive}{true \ positive + false \ negative}$$  

#### accuracy and cost-sensitive accuracy  

The ratio of examples whose labels (outcomes) are predicted correctly.
true positive+true negativeall cases
Cost-sensitive accuracy is a method to deal with the situation when different classes have varied importance. A cost (a positive number) will be assigned to both types of mistake: false positive and false negative to compute a cost-sensitive accuracy.  

#### area under the ROC curve (AUC)  
ROC curves use a combination of the recall (sensitivity) and false positive rate (1-specificity) to build up a summary figure of the classification performance.
$$sensitivity\overset{def}{=}\frac{true \ positive}{true \ positive + false \ negative}$$
$$specificity\overset{def}{=}\frac{true \ negative}{true \ negative + false \ positive}$$


#### Cross-validation  

The process for tuning hyperparameters used especially when there’s few training examples.
It can be summarized into four steps: 1) fix the values of the hyperparameters for the methods you chose; 2) randomly split the training set into several subsets of the same size, which is called fold (e.g., F1,F2,F3,F4,F5 denotes five folds and each Fk contains 20% of your training data); 3)train five models and four folds for each model (e.g., Model 1 f1 will be trained based on F2,F3,F4,F5 and F1 will be validation set ); 4) apply the average values of each iteration as the final value. 

#### Bootstrap Aggregation (Bagging)  

Bootstrap aggregation, or bagging, is an ensemble learning approach that generally reduces the variance and overfitting of an unstable algorithm. The approach includes drawing a number of bootstrap resamples (random samples with replacement) from the training dataset, followed by applying the algorithm of interest to each of the bootstrap resamples. Two approaches for bagging include: 1) selecting the class with the most “votes” or 2) averaging over the classifiers. However, there is a tradeoff to bagging. Models that are most helped by bagging, such as decision trees, are often those which are interpretable, however, this interpretability element is lost in the bagging process.

#### Boosting  

Boosting is another form of ensemble learning that leverages a combination of weak learners, which are simple classifiers that perform only slightly better than random guessing. It is a repetitive process, where the last prediction changes the weight for future predictions. The key idea is that these weak learners will learn from the previously misclassified training examples to improve the overall performance of the ensemble. 
Black Box Algorithm
A ‘black box’ is used to describe algorithms that cannot be easily tested or analyzed. The inner workings of these algorithms are therefore not easily interpretable. This contrasts models commonly used in health science research, such as regression models, where coefficients can be referenced to understand the model outputs.
#### Stacking [ashley send]  

#### Trees and Decision trees  

A decision tree is an acyclic graph and drawn upside down with its roots at the top. It includes two main entities, nodes and branches. In each node of the tree, a specific covariate is examined (i.e., whether this patient has diabetes or not). If the value of the covariate is below a specific threshold, then the left branch is followed; otherwise, the right branch is followed. The decision of the class assignment (i.e., whether this patient will die or not) is made when reaching the leaf note. Classification trees and regression trees are applied for categorical outcomes and continuous outcomes, respectively. 

#### Nearest neighbor method  

Nearest neighbor methods assign to an unclassified sample point the classification of the nearest of a set of previously classified points. The distance between the unclassified points and previously classified points can be measured using different distance metrics, including Euclidean distance, Manhattan distance,  Minkowski distance, and Hamming distance
K-nearest neighbors algorithm, also known as KNN or K-NN is one of nearst neighbor methods. The k value in the k-NN algorithm defines how many neighbors will be checked to determine the classification of a specific query point. For example, if k=1, the instance will be assigned to the same class as its single nearst neighbor. Defining K is an example of bias variance tradeoff, as different values can lead to problems of overfitting or underfitting. 

#### Neural Networks  
Neural Networks are a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that reflect the hehavior of human brain. Neural Networks are composed of an input layers, one or more hidden layers, and an outputlayer. Each node on layers connects to another nad has an associated weight and threshold to determine whether the data is passed along to the next layer of network.
Machine learning algorithms that use neural networks generally do not need to be programmed with specific rules that define what to expect from the input. The neural net learning algorithm instead learns from processing many labeled examples (i.e. data with with "answers") that are supplied during training and using this answer key to learn what characteristics of the input are needed to construct the correct output.



#### Parametric vs Nonparametric vs Semiparametric [ashley write, maybe include explanation of saturated model]
#### decision boundary  [maybe better idea would be to introduce decision theory ML versus information theory ML]
The boundary separating the examples of different classes.  

#### introduce decision theory ML versus information theory ML [ashley write]

#### Feature Engineering
In the same way that we cannot take a raw dataset and immediately begin a standard regression analysis, we cannot take raw data and immediately apply it to our machine learning algorithm. The process of turning the raw data we begin with into a workable dataset in machine learning is known as ‘feature engineering’. Many of these processes should sound familiar to epidemiologists. 
A feature is analogous to covariate, usually denoted as  x(j). Feature vectors are analogous to observation or record in Epidemiology, usually denoted as Xi. i and j are the index for the position. The feature at position j in all the feature vectors always contains the same kind of information in on dataset, for instance, x1(2) and x2(2) indicated the second feature (covariate) for observation 1 and observation 2, respectively. A single dataset can have many features within a single feature vector. 
For example, if we are interested in predicting the risk of preterm birth among women in a given healthcare system, we could extract important data points from their electronic medical records. Feature selection should be based on a priori knowledge, in the same way that subject matter expertise is essential to understanding which covariates should be included in a regression analysis. We may be interested in the effect of maternal age on preterm birth. We would then extract the mother’s date of birth from the electronic health record to create a feature called maternal age. In this case, the feature vector for maternal age would consist of a single observation per patient. Highly informative features are ones that do a good job of predicting the label.
There are some learning algorithms that will only handle numerical data. The process of transforming a categorical feature into dummy variables is known as one-hot encoding. For example, our electronic health record may have a checklist of chronic health conditions the patient has been diagnosed with (hypertension, diabetes, depression). For a patient who has all of the conditions, the process of one-hot encoding would transform this list into a vector of numerical values:

diabetes=[1,0,0] hypertension=[0,1,0] depression=[0,0,1] 

An alternative feature engineering approach, known as binning, takes a continuous variable and categorizes it into multiple binary features called bins or buckets. For example, we may be interested in how categories of maternal BMI predict preterm birth. We would take our continuous maternal BMI data point and transform it into four bins. All values of BMI <18.5 are added to the “underweight” bin,  18.5 to 25 to the “normal weight” bin, 25 to 30 in the “overweight”, and values >30 in the “obese” bin. 
One way to improve the speed of learning is through normalization. This type of feature engineering transforms the feature into a standard range of values. Typically this is between the interval [-1, 1] or [0,1]. The general formula for normalization is:
x(j)=x(j)-min(j)max(j)-min(j), where min(j)and max(j) represent the minimum and maximum value of the original feature, respectively. Normalization is not a necessary component of feature engineering; however, it is used 

### Standardization 
The process of rescaling the feature values so that they have the properties of a standard normal distribution (XN(0,1)).
