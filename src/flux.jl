import Flux

struct Linear{Ts}
    W::Ts
end
export Linear
function Linear(in::Int, out::Int; 
                T = Float64, initW = (x...) -> zeros(T, x...))
    Linear(Flux.param(initW(out, in)))
end
(a::Linear)(x) = a.W * x
Flux.treelike(Linear)

Base.show(io::IO, l::Linear) = print(io, "Linear( $(size(l.W, 2)), $(size(l.W, 1)))")

function fluxreconstruct(elem, w, i)
    fs = []
    for fn in fieldnames(elem)
        f = getfield(elem, fn)
        if typeof(f) <: Flux.TrackedArray
            if size(f) != size(w[i])
                error("$(f) and $(w[i]) have different size")
            end
            push!(fs, Flux.TrackedArray(w[i]))
            i += 1
        else
            push!(fs, f)
        end
    end
    typeof(elem)(fs...), i
end
function fluxreconstruct(chain::Flux.Chain, w)
    i = 1
    elems = []
    for elem in chain.layers
        newelem, i = fluxreconstruct(elem, w, i)
        push!(elems, newelem)
    end
    Flux.Chain(elems...), i
end
