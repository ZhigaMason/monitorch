## Issues

:)

## Research

### Software

- [x] Tensorboard
- [x] PyTorch hooks
- [x] nn.Module `children` method
- [x] Tests (what type of tests)

### DevOps
- [x] pyproject.toml
- [ ] publishing at anaconda
- [ ] CI/CDing
- [x] Autodocumentation

## Architecture

- [x] 'Ends communication channels (injections/calls/intermediate objects)
- [-] Class hierarchies

## Optimizer (?)

- [ ] Optimizer specific gradient shenenigans (i.e. aggregated velocity, acceleration vectors, nesterov action smth)

## Backend (metadata collection - gatherers)

- [x] Backward callback
- [x] Feedforward callback
- [ ] Attention callback
    - [ ] Forward
    - [ ] Backward
- [ ] Recurent callback
    - [ ] Forward
    - [ ] Backward

## Middleend (metadata parsing - preprocessors)

- [x] fix naming convention for running/memory (inplace/inmemory)
- [x] neuron activation and death
    - [-] spatial attention ~~and channels~~
    - [x] running mean & variance
        - min, max
    - [x] all data accumulated
        - median, IQR
- [x] output norms
    - [x] inplace
    - [x] inmemory
- [x] gradient norms
    - [x] weights vs bias vs outputs (vs inputs? [they are not used in backprop essentially])
    - [x] running mean & variance
        - [x] min, max
    - [x] all data accumulated
        - median, IQR
    - [ ] gradient death & activation
        - [x] abstract activation compututations into monitorch.numerical
- [x] utility running statistical info
- [x] loss observers
- [ ] net-aggregation preprocessors
    - I think it makes more sense to inject such functionality into lenses, as they have an overview over all layers

## Frontend (metadata visualisation)

- plot hierarchy
    - Tag - defined by lens
        - Tensorboard - use tag and dump all plots
        - Matplotlib  - group into one subfigure (with shared x if x is epoch)

- generic vizualizer methods
    - statistics & log statistics
        - mean, min, max, median
        - variance, iqr
    - probability
    - multivariate relations
        - stack graph
        - multiline graph

- [ ] Matplotib (static)
- [x] ~~Matplotib (dynamic? [it is not designed for that])~~
- [x] Tensorboard
    - [-] How to plot ranges - additional line plots


## Documentation

- [ ] Docstrings
- [ ] autodocumnetation
