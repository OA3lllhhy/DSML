Step of using minuit to do parameter estimation

*Step 1*: Data preparation

```python
import pandas as pd

data = pandas.read_csv("data/xxx.csv")

x_values = data['x'].values
y_values = data['y'].values
y_err = data['y_err'].values
```

*Step 2*: Define the model function you need to fit with data

```python
import numpy as np
def model_func(x, a, b, c):
    return a * np.exp(-(x - b) / c)
```

*Step 3*: Define the cost function. (e.g. the leastsquare function)

```python
from iminuit.cost import LeastSquares

# Initialize the cost function
least_square_cf = LeastSquares(x_values, y_values, y_error, model_func)
```

*Step 4*: Define initial guess of parameter and the Minuit object, perform the fit
```python
from iminuit import Minuit

a_est =
b_est = 
c_est =
initial_param = [a_est, b_est, c_est]

# Define the minuit object
mobj = Minuit(least_square_cf, *initial_param)

# Perform the fit
mobj.migrad()
mobj.hesse()

# print fitted parameters
for param in mobj_ex1_p2.params:
    print('{} = {:.2f} +/- {:.2f}'.format(param.name, param.value, param.error))

# Plot the result
# ---------------
```

*Small tricks*:
```python
# This return the minimum chi-square value
mobj.fmin.fval

# This return the degree of freedom
mobj.fmin.ndof

# This return the reduced chi-square
mobj.fmin.reduced_chi2
```