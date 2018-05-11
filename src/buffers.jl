struct Buffer{Ts, Ta}
    states::CircularBuffer{Ts}
    actions::CircularBuffer{Ta}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
export Buffer
function Buffer(; statetype = Int64, actiontype = Int64, 
                  capacity = 2, capacitystates = capacity,
                  capacityrewards = capacity - 1)
    Buffer(CircularBuffer{statetype}(capacitystates),
           CircularBuffer{actiontype}(capacitystates),
           CircularBuffer{Float64}(capacityrewards),
           CircularBuffer{Bool}(capacityrewards))
end
function pushstateaction!(b, s, a)
    pushstate!(b, s)
    pushaction!(b, a)
end
pushstate!(b, s) = push!(b.states, deepcopy(s))
pushaction!(b, a) = push!(b.actions, a)
function pushreturn!(b, r, done)
    push!(b.rewards, r)
    push!(b.done, done)
end

struct EpisodeBuffer{Ts, Ta}
    states::Array{Ts, 1}
    actions::Array{Ta, 1}
    rewards::Array{Float64, 1}
    done::Array{Bool, 1}
end
EpisodeBuffer(; statetype = Int64, actiontype = Int64) = 
    EpisodeBuffer(statetype[], actiontype[], Float64[], Bool[])
function pushreturn!(b::EpisodeBuffer, r, done)
    if length(b.done) > 0 && b.done[end]
        s = b.states[end]; a = b.actions[end]
        empty!(b.states); empty!(b.actions); empty!(b.rewards); empty!(b.done)
        push!(b.states, s)
        push!(b.actions, a)
    end
    push!(b.rewards, r)
    push!(b.done, done)
end
export EpisodeBuffer

struct AdvantageBuffer{Tg, Ta}
    pg::CircularBuffer{Tg}
    actions::CircularBuffer{Ta}
    values::CircularBuffer{Float64}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
export AdvantageBuffer
function AdvantageBuffer(; gradienttype = Array{Any, 1}, actiontype = Int64, 
                           nsteps = 1, capacity = nsteps + 1, γ = .9)
    AdvantageBuffer(CircularBuffer{gradienttype}(capacity),
                    CircularBuffer{actiontype}(capacity),
                    CircularBuffer{Float64}(capacity),
                    CircularBuffer{Float64}(capacity - 1),
                    CircularBuffer{Bool}(capacity - 1))
end
pushstateaction!(b::AdvantageBuffer, s, a) = push!(b.actions, a)



mutable struct ArrayCircularBuffer{T}
    data::T
    capacity::Int64
    start::Int64
    counter::Int64
    full::Bool
end
function ArrayCircularBuffer(arraytype, datatype, elemshape, capacity)
    ArrayCircularBuffer(arraytype(zeros(datatype, elemshape..., capacity)),
                        capacity, 0, 0, false)
end
import Base.push!, Base.view, Base.endof, Base.getindex
for N in 2:5
    @eval current_module() begin
        function push!(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, x) where T
            setindex!(a.data, x, $(fill(Colon(), N-1)...), a.counter + 1)
            a.counter += 1
            a.counter = a.counter % a.capacity
            if a.full 
                a.start += 1 
                a.start = a.start % a.capacity
            end
            if a.counter == 0 a.full = true end
            a.data
        end
    end
    for func in [:view, :getindex]
        @eval current_module() begin
            @inline function $func(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, i) where T
                idx = (a.start + i - 1) .% a.capacity + 1
                $func(a.data, $(fill(Colon(), N-1)...), idx)
            end
            @inline function $func(a::ArrayCircularBuffer{<:AbstractArray{T, $N}}, i, nmarkov) where T
                nmarkov == 1 && return $func(a, i)
                numi = typeof(i) <: Number ? 1 : length(i)
                idx = zeros(Int64, numi*nmarkov)
                c = 1
                for j in i
                    for k in j - nmarkov + 1:j
                        idx[c] = (a.start + k - 1) % a.capacity + 1
                        c += 1
                    end
                end
                res = $func(a.data, $(fill(Colon(), N-1)...), idx)
                s = size(res)
                reshape(res, $([:(s[$x]) for x in 1:N-2]...), nmarkov, numi)
            end
        end
    end
end
endof(a::ArrayCircularBuffer) = a.full ? a.capacity : a.counter

struct ArrayStateBuffer{Ts, Ta}
    states::ArrayCircularBuffer{Ts}
    actions::CircularBuffer{Ta}
    rewards::CircularBuffer{Float64}
    done::CircularBuffer{Bool}
end
export ArrayStateBuffer
function ArrayStateBuffer(; arraytype = Array, datatype = Float64, 
                            elemshape = (1), actiontype = Int64, 
                            capacity = 2, capacitystates = capacity,
                            capacityrewards = capacity - 1)
    ArrayStateBuffer(ArrayCircularBuffer(arraytype, datatype, elemshape, 
                                         capacitystates),
                     CircularBuffer{actiontype}(capacitystates),
                     CircularBuffer{Float64}(capacityrewards),
                     CircularBuffer{Bool}(capacityrewards))
end
pushstate!(b::ArrayStateBuffer, s) = push!(b.states, s)

import DataStructures.isfull
isfull(b::Union{Buffer, AdvantageBuffer, ArrayStateBuffer}) = isfull(b.rewards)
