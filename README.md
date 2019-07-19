# Multi-Modal

## Dependencies
* NumPy
* Visdom
* PyTorch



## Running the Code
**Terminal 1:**
Start a local Visdom server in isolation with the command `visdom`

**Terminal 2:**
`cd` into the repo and then run `python train.py {script_name}`, where the script can be any file in the `scripts` directory. You don't have to include the `.py` at the end of the script name, but it won't break if you do.



## Default Scripts
* **bellman**
Forms an approximation of the value distribution Z in the form of the distributional Bellman equation `Z(s, a) = r + γ Z(s', a')`, then fits a Gaussian mixture distribution to `Z` by minimizing cross entropy

* **gridworld:** Approximates reward distributions for each non-terminal state-action pair in a simple discrete environment. It's not actually a grid world so I don't know why I named it that

* **dist_ops:** Performs a few sample scalar operations on a Gaussian mixture to prove that it works

* **fit_dist:** Fits one Gaussian mixture `Q` to another distribution 'P'. To fit it quickly, I minimize the mean squared error loss between `P(x)` and `Q(x)`, where `x ~ P`. It can also be fit by minimizing cross entropy (which is much more feasible if `P` is unknown), so uncomment that line in the script to see it train that way instead. It's much slower, but it still works
