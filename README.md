# cs4563-machine-learning

## Overview

There are many factors involved when a person is making his/her job decision. Companies are interested
in knowing whether their job candidates are willing to work for the job so they can better allocate their
resources. As students, we also want to know what we should take account in our decision making process.
The data set we use is from [this kaggle site](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists)
on the job of change of data scientists.


## Data Processing

### Inputs & Outputs
From our data source, we know that we have 14 features for our training data, of which 12 of them will
be used as inputs and one will be used as the target (output). For the inputs, we have:

- City: City code
- City_development_index: Development index of the city (scaled)
- Gender: Gender of the job candidate
- Relevant_experience: Relevant experience of candidate
- Enrolled_university: Type of University that the candidate has enrolled (if any)
- Education_level: Education level of the candidate
- Major_discipline: Education major discipline of the candidate
- Experience: Candidate total experience in years
- Company_size: Number of employees in current employer’s company
- Company_type: Type of the current employer’s company
- Last_new_job: Different in years between previous job and current job (how many years the candidate
- stayed in the last company)
- Trianing_hours: Training hours completed by the candidate for current job

And for the target output, we have not been looking for job change (target equals zero) and looking for
a job change (target equals one).

### Data Classification
To begin with, what we need to do to process data is to correctly classify them. The text values of
the categorical features are replaced with preset numerical values as shown in the following tables, which
include gender, relevant experience, enrolled university, education level, major disciplines, company sizes
and types. We also reformat some categorical features to better classify them.

### Deal with Null Values
What’s more, we need to deal with null values in the dataset. Many data we have are with null values.
There are a couple of ways we can go with here. One intuitive approach is to add another group (new
classification category) such as Unclassified. However, it doesn’t give us any useful information. If we
see any correlation with this new category, we couldn’t interpret it correctly due to its unknown identity.
We only add a new category Not Applied to Major_discipline, because major disciplines don’t apply to candidates
that have high school or lower education. Thus we can choose either to interpret missing data or drop the data
that contains null values. As we have about 20000 data points, we can afford using a subsample of the entire
dataset. We tried three ways: first, use median for all data (classify them to the average group); secondly,
find K-Nearest Neighbors and classify the missing value; what’s more, drop all data that contains null values.

### Deal with Imbalanced Data
Additionally, the dataset we are working with is a very imbalanced data. We have about 25 percent of our
data with target being 1, and 75 percent of our data with target being 0. After dropping data containing null
values, the distribution skews to about 16 percent of data with target being 1 and 84 percent of data with
targeting being 0. There are also three approaches we used to deal with this imbalanced distribution. One is
do nothing - use the natural distribution. This approach is bad, because based on original data simply predicting
all the results to 0 would give us 75 percent accuracy (84 percent accuracy on the data after dropping null
values). The other two approaches to balance the datasets are similar: oversampling the minority class by
synthesizing the new minority class and undersampling the majority class by adjusting the class weight. The
method we use to synthesize the new minority class is SMOTE, a synthetic minority oversampling technique.

### Results of Data Processing
Following are the results obtained after data processing. We can see that out of three ways dealing with null
values, simply dropping all null values performs the best. This may because we have many null values to interpret
and the process of interpretation would introduce more errors. Since we also have enough data after dropping all
null values, we would go with just dropping all null values for the future steps. 

To better evaluate the performance of SMOTE and reweighting the data, we perform both of them on our logistic
regression model and SVM models. Here are the results for Logistic Linear and Logistic Poly with Ridge
regularization and SVM with Linear and Poly Kernel using data after dropping all null values.

Finally, our strategy for training the models is clear. We would use data after dropping null values as our
dataset for the training process. Because neural networks are not subject for weight changing, for logistic
regression and neural network models, we would use SMOTE to oversample the minority class in training data. For
SVM models, we would use reweight the data since it performs slightly better. We are also curious to see how
accurate they would be with different methods of data processing.

## Algorithms Analysis
### Unsupervised Learning (K-cluster)
First of all, we tried K-Means clustering on our preprocessed data to find out whether there are interesting
structures. The library we used is sklearn.cluster.KMeans. We first split the data to 70% training and 30%
testing. Since the result has two classes, we set the n_clusters to be 2. Test accuracy is based on the
probability that X_test is correctly classified. We run the function 100 times and each time with different
random states to collect the best testing accuracy. The max test accuracy is 78.79%.

Since the data has 11 features, we used a method called t-distributed Stochastic Neighbor Embedding in
sklearn.manifold.TSNE to lower the dimension of data to 2. It can convert similarities between data points
to joint probabilities. After applying this technique, the result of K-Means clustering is shown in the
figure below, where the data are divided into two clusters and the red points are the cluster centers.

### Logistic Regression
The first supervised learning method we choose is logistic regression, including logistic regression with
lasso and ridge regularization, and also logistic regression using polynomial feature transformation with
lasso and ridge regularization. We chose SMOTE to handle the imbalance data since it gave better test accuracy.
Same as k-cluster, the data was splitted into 70% training and 30% testing. The regularization penalties we
tried are 0.001,0.01,0.1,1,10 and 100. 

After running through all the logistic models, the logistic regression using polynomial feature transformation
with lasso and ridge both give the best test accuracy at 86.11% with lambda equals 0.1.

### SVM
The second supervised learning method we choose is SVM, including linear SVM using sklearn.svm. SVC linear kernel,
Radial Basis Function SVM, Polynomial SVM and linear SVM using lasso and ridge regularization with
sklearn.svm.LinearSVC. We chose to reweight our features instead of SMOTE since it gave higher test accuracy in
this case. The data was splitted into 70% training and 30% testing. The regularization penalties we tried are
0.001,0.01,0.1,1,10 and 100.

The best test accuracy among these models is 84.68% with Polynomial SVM with lambda equals 100. The training
accuracy in this case is 84.34%

### Neural Network
The third supervised learning method we choose is Neural Network, including sigmoid, tanh and relu as activation
function. We chose to reweight our features instead of SMOTE since the sklearn neural network does not support
reweight. There is one hidden layer with 8 neurons since it gave the best result. The data was splitted into 70%
training and 30% testing. The regularization penalties we tried are 0.001,0.01,0.1,1,10  and 100. The learning rate
we used are 0.001,0.01,0.1,1,10 and 100.

Using sigmoid as activation function with lambda equals 10 and learning rate equals 0.01 gives the best test
accuracy 85.95%.
