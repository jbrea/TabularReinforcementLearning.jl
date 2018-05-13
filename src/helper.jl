@inline function maximumbelowInf(values)
    m = -Inf64
    for v in values
        if v < Inf64 && v > m
            m = v
        end
    end
    if m == -Inf64
        Inf64
    else
        m
    end
end

macro subtypes(supertype, body, subtypes...)
    for subtype in subtypes
        @eval (mutable struct $subtype <: $supertype
            $body
        end;
        export $subtype)
    end
end

@inline getvalue(params, state::Int) = params[:, state]
@inline getvalue(params::Vector, state::Int) = params[state]
@inline getvalue(params, action::Int, state::Int) = params[action, state]
@inline getvalue(params, state::AbstractArray) = params * state
@inline getvalue(params::Vector, state::Vector) = dot(params, state)
@inline getvalue(params, action::Int, state::AbstractArray) = 
    dot(view(params, action, :), state)
