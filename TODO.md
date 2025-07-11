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
- [-] CI/CDing
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
- [x] restructure inplace/inmemory
- [x] resolve abstract backward/foreward hook name collision
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
    - [x] gradient death & activation
        - [x] abstract activation compututations into monitorch.numerical
- [x] utility running statistical info
- [x] loss observers
- [x] learnable parameter norm

## Lens

- inplace and inmemory variants

- [x] interaction with inspector
    - [x] hooking to modules
    - [x] interactions with vizualizer
- [-] net aggregation

- [ ] Debug Lens
- [x] Gradient Geometry
- [x] Gradient Activation
- [x] Output Activation
- [x] Output Norm
- [x] Parameter norms
- [x] Loss/Metrics

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

- [x] Matplotib (static)
    - [x] \(pre\)allocating figures and axis
    - [x] axis metainfo (logy, shadows)
    - [x] generic methods
- [x] ~~Matplotib (dynamic? [it is not designed for that])~~
- [x] Tensorboard
    - [-] How to plot ranges - additional line plots


## Documentation

- [ ] Docstrings
- [ ] autodocumnetation
