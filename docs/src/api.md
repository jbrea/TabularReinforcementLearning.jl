# API

New learners, policies, callbacks, environments, evaluation metrics or stopping
criteria need to implement the following functions.

## Learners
```@docs
update!(learner, buffer)
```

```@docs
selectaction(learner, policy, state)
```

## Policies
```@docs
selectaction(policy, values)
```

```@docs
getactionprobabilities(policy, state)
```

## Callbacks
```@docs
callback!(callback, rlsetup, state, action, reward, done)
```

## [Environments](@id api_environments)
```@docs
interact!(action, environment)
```

```@docs
getstate(environment)
```

```@docs
reset!(environment)
```

## Evaluation Metrics

```@docs
getvalue(metric)
```

## Stopping Criteria
```@docs
isbreak!(stoppingcriterion, state, action, reward, done)
```
