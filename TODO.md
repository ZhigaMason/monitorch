# Refactor

- [ ] Reduce code duplication in lens' and preprocessor' classes (by abstracting things away) => ease of custom lens definition
    - [ ] As a subpoint, concretize value = self.state (self.\_value) processors
    - [ ] tie geometry computation accross all different fields
    - [ ] Concretize what should be coupled
- [ ] id/name identification in the "back-end"
- [ ] no\_grad global state vs model.eval() local state in preprocessors that skip no\_grad passes explicitly

# Features

- [ ] Optimizer parameters (ie lr, moment-coefs)
- [ ] Log Optimizer Update Step
    - [ ] Gatherer
        - [ ] Should transform `optimizer.state_dict()['state']` to some more identifiable form (see refactor id/name)
        - [ ] Must pass optimizer parameters to preprocessor
    - [ ] Preprocessor
        - [ ] Must transform billion parameters to statistics guided by Optimizer class and its parameters (callback to get transformation function is a must have)
        - [ ] Geometry of Update Step
    - [ ] Lens
- [ ] MLflow visualisations
    - [ ] decide whether to inject monitorch into autolog ecosystem or create a separate MLflow Visualizer

# Debug

- [ ] Check on PyTorch Lightning
- [ ] Chck on DataParallel / DistributedDataParallel
- [ ] Reevaluate tests

# Optimize

- [ ] Measure in-place / in-memory operations
- [ ] Measure scalability
