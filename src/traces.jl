"""
    struct NoTraces <: AbstractTraces

No eligibility traces, i.e. ``e(a, s) = 1`` for current action ``a`` and state
``s`` and zero otherwise.
"""
struct NoTraces <: AbstractTraces end
export NoTraces

# TODO: use sparse matrices for speed.
for kind in (:ReplacingTraces, :AccumulatingTraces)
    @eval (struct $kind{Tt} <: AbstractTraces
                λ::Float64
                γλ::Float64
                trace::Tt
                minimaltracevalue::Float64
            end;
            export $kind;
            function $kind(ns, na, λ::Float64, γ::Float64; 
                           minimaltracevalue = 1e-12,
                           trace = sparse([], [], [], na, ns))
                $kind(λ, γ*λ, trace, minimaltracevalue)
            end)
end
@doc """
    struct ReplacingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" ReplacingTraces
@doc """
    ReplacingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" ReplacingTraces()
@doc """
    struct AccumulatingTraces <: AbstractTraces
        λ::Float64
        γλ::Float64
        trace::Array{Float64, 2}
        minimaltracevalue::Float64

Decaying traces with factor γλ. 

Traces are updated according to ``e(a, s) ←  1 + e(a, s)`` for the current action-state
pair and ``e(a, s) ←  γλ e(a, s)`` for all other pairs unless
``e(a, s) < `` `minimaltracevalue` where the trace is set to 0 
(for computational efficiency).
""" AccumulatingTraces
@doc """
    AccumulatingTraces(ns, na, λ::Float64, γ::Float64; minimaltracevalue = 1e-12)
""" AccumulatingTraces()

function increasetrace!(traces::ReplacingTraces, state::Int, action)
    traces.trace[action, state] = 1.
end
function increasetrace!(traces::ReplacingTraces, state::Vector, action)
    @inbounds for i in find(state)
        traces.trace[action, i] = state[i]
    end
end
function increasetrace!(traces::ReplacingTraces, state::SparseVector, action)
    @inbounds for i in 1:length(state.nzind)
        traces.trace[action, state.nzind[i]] = state.nzval[i]
    end
end
function increasetrace!(traces::AccumulatingTraces, state::Int, action)
    traces.trace[action, state] += 1.
end
function increasetrace!(traces::AccumulatingTraces, state::Vector, action)
    @inbounds for i in find(state)
        traces.trace[action, i] += state[i]
    end
end
function increasetrace!(traces::AccumulatingTraces, state::SparseVector, action)
    @inbounds for i in 1:length(state.nzind)
        traces.trace[action, state.nzind[i]] += state.nzval[i]
    end
end


function discounttraces!(traces)
    BLAS.scale!(traces.γλ, traces.trace.nzval)
    if rand() < .01
        clamp!(traces.trace.nzval, traces.minimaltracevalue, Inf)
    end
end
resettraces!(traces) = BLAS.scale!(0., traces.trace)

function updatetraceandparams!(traces, params, factor)
    s = traces.trace
    c = 1
    @inbounds for k in 1:length(s.nzval)
        while s.colptr[c+1] - 1 < k || s.colptr[c] == s.colptr[c+1]; c += 1; end
        params[s.rowval[k], c] += factor * s.nzval[k]
    end
    discounttraces!(traces)
end

function updatetrace!(traces, state, action)
    discounttraces!(traces)
    increasetrace!(traces, state, action)
end
