__precompile__()

module TabularReinforcementLearning

using DataStructures, Parameters
include("helper.jl")
include("forced.jl")
include("buffers.jl")
include("traces.jl")
include("epsilongreedypolicies.jl")
include("softmaxpolicy.jl")
include("mdp.jl")
include("metrics.jl")
include("stoppingcriterion.jl")
include("callbacks.jl")
include("preprocessor.jl")
include("flux.jl")
include("learner/tdlearning.jl")
include("learner/prioritizedsweeping.jl")
include("learner/policygradientlearning.jl")
include("learner/mdplearner.jl")
include("learner/montecarlo.jl")
include("learner/deepactorcritic.jl")
include("learner/dqn.jl")
include("rlsetup.jl")
include("learn.jl")
# include("comparisontools.jl")
    

end # module
