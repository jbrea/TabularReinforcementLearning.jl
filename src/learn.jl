@inline function step!(rlsetup, a)
    @unpack learner, policy, buffer, preprocessor, environment, fillbuffer = rlsetup
    s0, r0, done0 = interact!(a, environment)
    s, r, done = preprocess(preprocessor, s0, r0, done0)
    if fillbuffer; pushreturn!(buffer, r, done) end
    if done
        s0 = reset!(environment)
        s = preprocessstate(preprocessor, s0) 
    end
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
    learn!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) with learning.
"""
function learn!(rlsetup)
    @unpack learner, buffer, stoppingcriterion = rlsetup
    a = firststateaction!(rlsetup) #TODO: callbacks don't see first state action
    while true
        sraw, a, r, done = step!(rlsetup, a)
        if rlsetup.islearning; update!(learner, buffer); end
        for callback in rlsetup.callbacks
            callback!(callback, rlsetup, sraw, a, r, done)
        end
        if isbreak!(stoppingcriterion, sraw, a, r, done); break; end
    end
end
"""
    learn!(rlsetups::Array{<:RLSetup, 1})

Runs [`rlsetups`](@ref RLSetup) asynchronously with learning. See [`toasync`](@ref) for
constructing a list of rlsetups with a learner with shared parameters.
"""
function learn!(rlsetups::Array{<:RLSetup, 1})
    Threads.@threads for rlsetup in rlsetups
        learn!(rlsetup)
    end
end

"""
    run!(rlsetup)

Runs an [`rlsetup`](@ref RLSetup) without learning.
"""
function run!(rlsetup::RLSetup)
    @unpack islearning, fillbuffer = rlsetup
    rlsetup.islearning = false
    rlsetup.fillbuffer = false
    learn!(rlsetup)
    rlsetup.islearning = islearning
    rlsetup.fillbuffer = fillbuffer
end

export learn!, run!
