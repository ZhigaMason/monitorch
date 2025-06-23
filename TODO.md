## Issues

:)

## Research

### Software

- [x] Tensorboard
- [x] PyTorch hooks
- [x] nn.Module `children` method
- [ ] Tests (what type of tests)

### DevOps
- [x] pyproject.toml
- [ ] publishing at anaconda
- [ ] CI/CDing
- [ ] Autodocumentation

## Architecture

- [x] 'Ends communication channels (injections/calls/intermediate objects)
- [-] Class hierarchies

## Optimizer (?)

- [ ] Optimizer specific gradient shenenigans (i.e. aggregated velocity, acceleration vectors, nesterov action smth)

## Backend (metadata collection - gatherers)

- [x] Backward callback
- [x] Feedforward callback
- [ ] Attention (forward) callback
- [ ] Recursive (forward) callback

## Middleend (metadata parsing - preprocessors)

- [ ] neuron activation and death
    - [ ] running mean & variance
        - min, max
    - [ ] all data accumulated
        - median, IQR
- [x] gradient norms
    - [ ] weights vs bias vs outputs (vs inputs? [they are not used in backprop essentially])
    - [x] running mean & variance
        - [ ] min, max
    - [ ] all data accumulated
        - median, IQR
    - [ ] gradient death & activation
- [ ] output gradient preprocessors
- [x] utility running statistical info
- [x] loss observers
- [ ] net-aggregation preprocessors

## Frontend (metadata visualisation)

- plot hierarchy
    - Tag - defined by lens
        - Tensorboard - use tag and dump all plots
        - Matplotlib  - group into one subfigure (with shared x if x is epoch)

- [ ] generic vizualizer methods
    - [ ] plots registration
    - [ ] statistics & log statistics
        - mean, min, max, median
        - variance, iqr
    - [ ] probability
    - [ ] multivariate relations
        - stack graph
        - multiline graph

- [ ] Matplotib (static)
- [ ] Matplotib (dynamic? [it is not designed for that])
- [ ] Tensorboard
    - [ ] How to plot ranges


## Documentation

- [ ] Docstrings
- [ ] autodocumnetation
