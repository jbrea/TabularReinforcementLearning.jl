"""
    @with_kw mutable struct RLSetup{Tl,Tb,Tp,Tpp,Te,Ts}
        learner::Tl
        environment::Te
        stoppingcriterion::Ts
        preprocessor::Tpp = NoPreprocessor()
        buffer::Tb = defaultbuffer(learner, environment, preprocessor)
        policy::Tp = defaultpolicy(learner, buffer)
        callbacks::Array{Any, 1} = []
        islearning::Bool = true
        fillbuffer::Bool = islearning
"""
@with_kw mutable struct RLSetup{Tl,Tb,Tp,Tpp,Te,Ts}
    learner::Tl
    environment::Te
    stoppingcriterion::Ts
    preprocessor::Tpp = NoPreprocessor()
    buffer::Tb = defaultbuffer(learner, environment, preprocessor)
    policy::Tp = defaultpolicy(learner, buffer)
    callbacks::Array{Any, 1} = []
    islearning::Bool = true
    fillbuffer::Bool = islearning
end
export RLSetup
RLSetup(learner, env, stop; kargs...) = RLSetup(learner = learner,
                                                environment = env,
                                                stoppingcriterion = stop;
                                                kargs...)
defaultpolicy(learner, buffer) = EpsilonGreedyPolicy(.1)
function defaultbuffer(learner, env, preprocessor)
    capacity = :nsteps in fieldnames(learner) ? learner.nsteps + 1 : 2
    statetype = typeof(preprocessstate(preprocessor, getstate(env)[1]))
    if capacity < 0
        EpisodeBuffer(statetype = statetype)
    else
        Buffer(capacity = capacity, statetype = statetype)
    end
end

# TODO: only share params
function toasync(rlsetup, createenv, n)
    [reconstruct(deepcopy(rlsetup), learner = rlsetup.learner, 
                                    environment = createenv()) for i in 1:n]

end
export toasync
