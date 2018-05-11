"""
    mutable struct ReduceEpsilonPerEpisode
        ϵ0::Float64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after each episode.

In episode n, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerEpisode
    ϵ0::Float64
    counter::Int64
end
"""
    ReduceEpsilonPerEpisode()

Initialize callback.
"""
ReduceEpsilonPerEpisode() = ReduceEpsilonPerEpisode(0., 1)
function callback!(c::ReduceEpsilonPerEpisode, rlsetup, sraw, a, r, done)
    if done
        if c.counter == 1
            c.ϵ0 = rlsetup.policy.ϵ
        end
        c.counter += 1
        rlsetup.policy.ϵ = c.ϵ0 / c.counter
    end
end
export ReduceEpsilonPerEpisode

"""
    mutable struct ReduceEpsilonPerT
        ϵ0::Float64
        T::Int64
        n::Int64
        counter::Int64

Reduces ϵ of an [`EpsilonGreedyPolicy`](@ref) after every `T` steps.

After n * T steps, ϵ = ϵ0/n
"""
mutable struct ReduceEpsilonPerT
    ϵ0::Float64
    T::Int64
    n::Int64
    counter::Int64
end
"""
    ReduceEpsilonPerT()

Initialize callback.
"""
ReduceEpsilonPerT(T) = ReduceEpsilonPerT(0., T, 1, 1)
function callback!(c::ReduceEpsilonPerT, rlsetup, sraw, a, r, done)
    c.counter += 1
    if c.counter == c.T
        c.counter == 1
        if c.n == 1
            c.ϵ0 = rlsetup.policy.ϵ
        end
        c.n += 1
        rlsetup.policy.ϵ = c.ϵ0 / c.n
    end
end
export ReduceEpsilonPerT

mutable struct LinearDecreaseEpsilon
    start::Int64
    stop::Int64
    initval::Float64
    finalval::Float64
    t::Int64
    step::Float64
end
export LinearDecreaseEpsilon
function LinearDecreaseEpsilon(start, stop, initval, finalval)
    step = (finalval - initval)/(stop - start)
    LinearDecreaseEpsilon(start, stop, initval, finalval, 0, step)
end
@inline setepsilon(policy, val) = policy.ϵ = val
@inline incrementepsilon(policy, val) = policy.ϵ += val
function callback!(c::LinearDecreaseEpsilon, rlsetup, sraw, a, r, done)
    c.t += 1
    if c.t == 1 setepsilon(rlsetup.policy, c.initval)
    elseif c.t >= c.start && c.t < c.stop
        incrementepsilon(rlsetup.policy, c.step)
    else
        setepsilon(rlsetup.policy, c.finalval)
    end
end

@with_kw mutable struct Progress 
    steps::Int64 = 10
    laststopcountervalue::Int64 = 0
end
Progress(steps) = Progress(steps = steps)
export Progress
progressunit(stop::ConstantNumberSteps) = "steps"
progressunit(stop::ConstantNumberEpisodes) = "episodes"
function callback!(c::Progress, rlsetup, sraw, a, r, done)
    stop = rlsetup.stoppingcriterion
    if stop.counter != c.laststopcountervalue && stop.counter % div(stop.N, c.steps) == 0
        c.laststopcountervalue = stop.counter
        lastvaluestring = join([getlastvaluestring(c) for c in rlsetup.callbacks])
        if lastvaluestring != ""
            lastvaluestring = "latest " * lastvaluestring
        end
        info("$(now())\t $(lpad(stop.counter, 9))/$(stop.N) $(progressunit(stop))\t $lastvaluestring.")
    end
end

getlastvaluestring(c) = ""

mutable struct Episode
    t::Int64
    N::Int64
end
Episode(N) = Episode(0, N)
function step!(c::Episode, done)
    if done
        c.t += 1
        c.t % c.N == 0
    else
        false
    end
end
mutable struct Step
    t::Int64
    N::Int64
end
Step(N) = Step(0, N)
function step!(c::Step, done)
    c.t += 1
    c.t % c.N == 0
end

@with_kw mutable struct EvaluateGreedy{T,Tc,Tu}
    ingreedy::Bool = false
    callback::Tc
    rlsetupcallbacks::Array{Any, 1} = []
    rlsetuppolicy::Any = 1
    stoppingcriterion::T
    every::Tu = Episode(10)
    values::Array{Any, 1} = []
end

function callback!(c::EvaluateGreedy, rlsetup, sraw, a, r, done)
    if c.ingreedy
        callback!(c.callback, rlsetup, sraw, a, r, done)
        if isbreak!(c.stoppingcriterion, sraw, a, r, done)
            push!(c.values, getvalue(c.callback))
            c.ingreedy = false
            rlsetup.islearning = true
            rlsetup.fillbuffer = true
            rlsetup.callbacks = c.rlsetupcallbacks
            rlsetup.policy = c.rlsetuppolicy
        end
    end
    if !c.ingreedy && step!(c.every, done)
        c.ingreedy = true
        rlsetup.islearning = false
        rlsetup.fillbuffer = false
        c.rlsetupcallbacks = rlsetup.callbacks
        rlsetup.callbacks = [c]
        c.rlsetuppolicy = rlsetup.policy
        rlsetup.policy = greedypolicy(rlsetup.policy)
    end
end
getvalue(c::EvaluateGreedy) = c.values

export EvaluateGreedy, Step, Episode
greedypolicy(p::AbstractEpsilonGreedyPolicy) = EpsilonGreedyPolicy(0)
greedypolicy(p::SoftmaxPolicy) = SoftmaxPolicy(Inf)

import FileIO:save
@with_kw struct SaveLearner{T}
    every::T = Step(10^3)
    filename::String = tempname()
end
export SaveLearner
function callback!(c::SaveLearner, rlsetup, sraw, a, r, done)
    if step!(c.every, done)
        save(c.filename * "_$(c.every.t).jld2", 
             Dict("learner" => rlsetup.learner))
    end
end
