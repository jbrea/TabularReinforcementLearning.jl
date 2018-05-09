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
@inlinesetepsilon(policy, val) = policy.ϵ = val
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
