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
function callback!(c::LinearDecreaseEpsilon, rlsetup, sraw, a, r, done)
    c.t += 1
    if c.t == 1 rlsetup.policy.ϵ = c.initval
    elseif c.t >= c.start && c.t < c.stop
        rlsetup.policy.ϵ += c.step
    else
        rlsetup.policy.ϵ = c.finalval
    end
end

struct Progress 
    steps::Int64
end
export Progress
showprogress(c, stop::ConstantNumberSteps) = stop.counter % div(stop.T, c.steps) == 0
showprogress(c, stop::ConstantNumberEpisodes) = stop.counter % div(stop.N, c.steps) == 0
function callback!(c::Progress, rlsetup, sraw, a, r, done)
    stop = rlsetup.stoppingcriterion
    if showprogress(c, stop)
        lastvaluestring = join([getlastvaluestring(c) for c in rlsetup.callbacks])
        info("$(now())\t $(stop.counter)/$(stop.T)\t $lastvaluestring.")
    end
end

getlastvaluestring(c) = ""
