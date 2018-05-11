@inline getvalue(params, state::Int) = params[:, state]
@inline getvalue(params::Vector, state::Int) = params[state]
@inline getvalue(params, action::Int, state::Int) = params[action, state]
@inline getvalue(params, state::AbstractArray) = params * state
@inline getvalue(params::Vector, state::Vector) = dot(params, state)
@inline getvalue(params, action::Int, state::AbstractArray) = 
    dot(view(params, action, :), state)

@inline function selectaction(learner::Union{TDLearner, AbstractPolicyGradient}, 
                              policy,
                              state)
    selectaction(policy, getvalue(learner.params, state))
end
@inline function selectaction(learner::Union{SmallBackups, MonteCarlo}, 
                              policy,
                              state)
    selectaction(policy, getvalue(learner.Q, state))
end
@inline function selectaction(learner::MDPLearner, 
                              policy::AbstractEpsilonGreedyPolicy, state)
    if rand() < policy.ϵ
        rand(1:learner.mdp.na)
    else
        learner.policy[state]
    end
end


"""
    learn!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
@inline function step!(rlsetup, a)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    s0, r0, done0 = interact!(a, environment)
    s, r, done = preprocess(preprocessor, s0, r0, done0)
    if fillbuffer; pushreturn!(buffer, r, done) end
    if done; s = preprocessstate(preprocessor, reset!(environment)) end
    if fillbuffer; pushstate!(buffer, s) end
    a = selectaction(learner, policy, s)
    if fillbuffer pushaction!(buffer, a) end
    s0, a, r, done
end
@inline function firststateaction!(rlsetup)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    if isempty(buffer.actions)
        sraw, done = getstate(environment)
        if done; sraw = reset!(environment); end
        s = preprocessstate(preprocessor, sraw)
        if fillbuffer; pushstate!(buffer, s) end
        a = selectaction(learner, policy, s)
        if fillbuffer; pushaction!(buffer, a) end
        a
    else
        buffer.actions[end]
    end
end

"""
    run!(learner, policy, callback, environment, metric, stoppingcriterion)
"""
function learn!(rlsetup)
    @unpack learner, buffer, stoppingcriterion = rlsetup
    a = firststateaction!(rlsetup)
    while true
        sraw, a, r, done = step!(rlsetup, a)
        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, sraw, a, r, done)
        end
        if isbreak!(stoppingcriterion, sraw, a, r, done); break; end
    end
end
function learn!(rlsetups::Array{<:RLSetup, 1})
    Threads.@threads for rlsetup in rlsetups
        learn!(rlsetup)
    end
end
function run!(rlsetup::RLSetup)
    @unpack islearning, fillbuffer = rlsetup
    rlsetup.islearning = false
    rlsetup.fillbuffer = false
    learn!(rlsetup)
    rlsetup.islearning = islearning
    rlsetup.fillbuffer = fillbuffer
end

export learn!, run!
