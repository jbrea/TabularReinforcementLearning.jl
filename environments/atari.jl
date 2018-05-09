using ArcadeLearningEnvironment, Images
import DataStructures: CircularBuffer
import TabularReinforcementLearning
import Flux
const T = TabularReinforcementLearning
import T.interact!, T.getstate, T.reset!, T.preprocessstate, T.selectaction,
T.callback!

struct AtariEnv
    ale::Ptr{Void}
    screen::Array{UInt8, 1}
    getscreen::Function
end
import ArcadeLearningEnvironment.getScreen
function getScreen(p::Ptr, s::Array{Cuchar, 1})
    sraw = getScreen(p)
    for i in 1:length(s)
        s[i] =  sraw[i] .>> 1
    end
end
function AtariEnv(name; 
                  colorspace = "Grayscale",
                  frame_skip = 4,
                  color_averaging = true,
                  repeat_action_probability = 0.)
    if isfile(name)
        ale = ALE_new()
        loadROM(ale, name)
        setBool(ale, "color_averaging", color_averaging)
        setInt(ale, "frame_skip", Int32(frame_skip))
        setFloat(ale, "repeat_action_probability", 
                 Float32(repeat_action_probability))
    else
        error("ROM $name could not be found.")
    end
    if colorspace == "Grayscale"
        screen = Array{Cuchar}(210*160)
        getscreen = getScreenGrayscale
    elseif colorspace == "RGB"
        screen = Array{Cuchar}(3*210*160)
        getscreen = getScreenRGB
    elseif colorspace == "Raw"
        screen = Array{Cuchar}(210*160)
        getscreen = getScreen
    end
    AtariEnv(ale, screen, getscreen)
end
function interact!(a, env::AtariEnv)
    r = act(env.ale, Int32(a - 1))
    env.getscreen(env.ale, env.screen)
    env.screen, r, game_over(env.ale)
end
function getstate(env::AtariEnv)
    env.getscreen(env.ale, env.screen)
    env.screen, game_over(env.ale)
end
reset!(env::AtariEnv) = reset_game(env.ale)

struct AtariPreprocessor end
function preprocessstate(::AtariPreprocessor, s)
    Flux.gpu(reshape(Float16.(imresize(reshape(s, 160, 210), 84, 84)/256), 84, 84, 1))
end
function preprocessstate(p::AtariPreprocessor, ::Void)
    Flux.gpu(zeros(Float16, 84, 84, 1))
end

mutable struct AtariBPROST
    background::Array{UInt8, 1}
    lastbasicfeatures::Array{Int64, 1}
    t::Int64
    determinebackgrounduntil::Int64
    backgroundbuffer::Array{Array{UInt8, 1}, 1}
end
function AtariBPROST(; determinebackgrounduntil = 18000)
    AtariBPROST(zeros(UInt8, 160 * 210), Int64[], 0,
                determinebackgrounduntil, Array{UInt8, 1}[])
end
function basicfeatures(p, s)
    ϕ = Int64[]
    for i in 1:length(s)
        if s[i] == p.background[i] || s[i] == 0
            continue
        else
            y = div(i, 160)
            x = mod(i, 160)
            c = div(x, 10)
            r = div(y, 15)
            push!(ϕ, toindbasic(c, r, s[i]))
        end
    end
    unique(ϕ)
end
toindbasic(c, r, k) = 1 + c + 16 * r + 16 * 14 * (k - 1)
function fromindbasic(i)
    c = (i - 1) % 16
    rk = div(i - c - 1, 16)
    r = rk % 14
    k = div(rk - r, 14) + 1
    (c, r, k)
end
function toindbpros(k1, k2, cdiff, rdiff)
    if k2 < k1
        tmp = k1
        k1 = k2
        k2 = tmp
        cdiff = -cdiff
        rdiff = -rdiff
    end
    cdiff + 16 + 31 * (rdiff + 13) + 
    31 * 27 * ((k1 - 1) * 128 - div((k1 - 1)*(k1 - 2), 2) + k2 - k1)
end
function bprosfeatures(ϕ)
    ϕbpros = Int64[]
    for i in 1:length(ϕ)
        @inbounds for j in i+1:length(ϕ)
            c1, r1, k1 = fromindbasic(ϕ[i])
            c2, r2, k2 = fromindbasic(ϕ[j])
            push!(ϕbpros, toindbpros(k1, k2, c1 - c2, r1 - r2))
        end
    end
    unique(ϕbpros)
end
toindbprot(k1, k2, cdiff, rdiff) = cdiff + 16 + 31 * (rdiff + 13) + 
   31 * 27 * (k1 - 1) + 31 * 27 * 128 * (k2 - 1)
function bprotfeatures(ϕ1, ϕ2)
    ϕbprot = Int64[]
    for i in 1:length(ϕ1)
        @inbounds for j in 1:length(ϕ2)
            c1, r1, k1 = fromindbasic(ϕ1[i])
            c2, r2, k2 = fromindbasic(ϕ2[j])
            push!(ϕbprot, toindbprot(k1, k2, c1 - c2, r1 - r2))
        end
    end
    unique(ϕbprot)
end

preprocessstate(p::AtariBPROST, s::Void) = sparsevec([1], [0.], 20652352)
function preprocessstate(p::AtariBPROST, s)
    p.t += 1
    if p.t < p.determinebackgrounduntil
        push!(p.backgroundbuffer, deepcopy(s))
    elseif p.t == p.determinebackgrounduntil
        push!(p.backgroundbuffer, deepcopy(s))
        p.background = UInt8.(median(hcat(p.backgroundbuffer...), 2)[:])
        p.lastbasicfeatures = basicfeatures(p, s)
    else
        ϕ = basicfeatures(p, s)
        ϕbpros = bprosfeatures(ϕ)
        ϕbprot = bprotfeatures(p.lastbasicfeatures, ϕ)
        p.lastbasicfeatures = ϕ
        return sparsevec([ϕ; ϕbpros + 28672; ϕbprot + 6938944], 
                  ones(length(ϕ) + length(ϕbpros) + length(ϕbprot)), 20652352)
    end
    sparsevec([1], [0.], 20652352)
end
