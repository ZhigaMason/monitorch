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
    - [x] running mean & variance
        - min, max
    - [ ] all data accumulated
        - median, IQR
- [x] utility running statistical info
- [x] loss observers

## Frontend (metadata visualisation)

- [ ] generic vizualizer methods
    - [ ] statistics
        - mean, min, max, median
        - variance, iqr
    - [ ] fractions/probability

- [ ] Matplotib (static)
- [ ] Matplotib (dynamic)
- [ ] Tensorboard


## Documentation

- [ ] Docstrings
- [ ] autodocumnetation
