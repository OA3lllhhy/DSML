**Type-I error** (significance level): 
$$
\alpha = \int_{\text{critical region}}f(X) \mathrm{d}x
$$

**Type-II error** (power): probability that we would **not** reject the null hypothesis when the alternate hypothesis is true. Probability of rejecting null hypothesis when alternative is true.

$$
1-\beta = \int_{\text{critical region}}g(X)\mathrm{d}x=P(\text{reject}\ H_0|H_1\ \text{is true})
$$
The large the power, the better. When you are simulating about $H_0$ the power is $1 - \frac{\text{rejection}}{\text{total simulation}}$. while if you are simulating about the $H_1$, do not need to be subtrated from 1.

**ROC Curve and AUC score** ($1-\beta$ vs $\alpha$): AUC is the area under the ROC curve, the AUC score in the range [0,1], higher the AUC score, better the classifier performance.

X-axis: False positive rate ($\alpha$)

* **True positive (TP)**: Samples in the +1 class that are correctly identified as the +1 class
<br>

* **False positive (FP)**: Samples in the -1 class that are incorrectly identified as the +1 class
<br>

* **True negative (TN)**: Samples in the -1 class that are correctly identified as the -1 class
<br>

* **False negative (FN)**: Samples in the +1 class that are incorrectly identified as the -1 class
$$
\text{FPR}=\frac{\text{False Positive (FP)}}{\text{False Positive (FP)}+\text{True Negative (TN)}}
$$

Y-axis: Ture Positive Rate ($1-\beta$)
$$
\text{TPR}: \frac{\text{True Positive (TP)}}{\text{True Positive (TP)}+\text{False Negative (FN)}}
$$


**KS test:** compares a data sample with a reference probability distribution:
$$
D = \text{max}_{X}|F_{\text{data}}(X)-G(X)|
$$
$G(x)$ is the cdf, D is in the range [0,1], lower the D score indicate better goodness of fit. Intuitive think: the difference between CDF and actual CDF, lower the value means the gap between are small which means no seperation and hence better fit.

**Reduced $\chi ^2$:** Indicate the under-fitting and Over-fitting. Ideal value of 1, greater that 1 indicate under-fitting and vice versa.

**Accuracy:**
$$
\text{Accuracy}=\frac{N_{\text{correct}}}{N_{\text{total}}}
$$

where $N_{\text{correct}}$ denotes the number of correctly predicted test points and $N_{\text{total}}$ denotes the total number of test points.


**Cross Validation Score:** For a K-fold cross validation
$$
\text{CVS} = \frac{1}{K}\sum_{i=1}^{K}\text{Accuracy}_i
$$

* **Accuracy**: as we have seen already, this is the proportion of samples that are classified correctly
<br>


$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

<br>

* **Fall-out or False Positive Rate**: the proportion of points with $y$ = -1 that are incorrectly predicted as $\hat{y}$ = +1
<br>


$$\text{Fall-out} = \frac{\text{+1 predictions with true class -1}}{\text{Points with true class -1}} = \frac{FP}{FP + TN}$$
<br>

* **Precision**: the proportion of points with predicted to have class +1 that have true class +1 i.e. fraction of predictions of class +1 that are correct
<br>


$$\text{Precision} = \frac{\text{+1 class predictions with true class +1}}{\text{Predictions of class +1}} = \frac{TP}{TP + FP}$$

<br>

* **Recall or True Positive Rate**: fraction of points with true class +1 that are correctly identified as class +1
<br>

$$
\text{Recall} = \frac{\text{+1 class predictions with true class +1}}{\text{Points with true class +1}} = \frac{TP}{TP + FN}
$$

**Performance metrics**:

$$
\begin{align*}
\text{Mean absolute error (MAE)}:\qquad\qquad & \text{MAE} = \frac{1}{N}\sum_{i = 1}^N|y_i - \hat{y}_i| \\[10pt]
\text{Mean squared error (MSE)}:\qquad\qquad & \text{MSE} = \frac{1}{N}\sum_{i = 1}^N(y_i - \hat{y}_i)^2 \\[10pt]
\text{Root-mean squared error (RMSE)}:\qquad\qquad & \text{RMSE} = \sqrt{\frac{1}{N}\sum_{i = 1}^N(y_i - \hat{y}_i)^2} \\
\end{align*}
$$

**Coefficient of determination $R^2$:**
$$
R^2 = 1-\frac{\sum_{i = 1}^N(y_i - \hat{y}_i)^2}{\sum_{i = 1}^N (y_i - \bar{y})^2}
$$
$y_i$ is the target for the test point $X_i$, $\hat{y_i}$ is te model prediction for the test point $X_i$.

A perfect prediction has an $R^2$ value equal to 1.

**Gini Impurity:** is the probability of incorrectly labelling a randomly selected sample.
$$
G_i = 1 - \sum _k p^2_{i,k}
$$

$p_{i,k}$ is the probability of selecting a sample with clss $k$ if we select a random sample at node $i$.

**Out Of Bag Score (OOB):**
In random forest, is like trainning multiple decision trees on the subsets of the whole trainning datasets. The out of bag data which is outside the trainning subset is used to test the single classifier and this is the OOB. This is similar to the cross validation.