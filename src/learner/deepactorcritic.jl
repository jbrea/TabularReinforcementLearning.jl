@with_kw struct DeepActorCritic{Tnet, Tpl, Tplm, Tvl, ToptT, Topt}
    nh::Int64 = 4
    na::Int64 = 2
    γ::Float64 = .9
    nsteps::Int64 = 5
    net::Tnet
    policylayer::Tpl = Linear(nh, na)
    policynet::Tplm = Flux.Chain(Flux.mapleaves(Flux.Tracker.data, net),
                             Flux.mapleaves(Flux.Tracker.data, policylayer))
    valuelayer::Tvl = Linear(nh, 1)
    params::Array{Any, 1} = vcat(map(Flux.params, [net, policylayer, valuelayer])...)
    opttype::ToptT = Flux.ADAM
    opt::Topt = opttype(params)
    αcritic::Float64 = .1
end
export DeepActorCritic
defaultpolicy(::DeepActorCritic, b) = SoftmaxPolicy1()

@inline function selectaction(learner::DeepActorCritic, policy, state)
    selectaction(policy, learner.policynet(state))
end
function update!(learner::DeepActorCritic, b)
    !isfull(b) && return
    h1 = learner.net(b.states[1])
    p1 = learner.policylayer(h1)
    v1 = learner.valuelayer(h1)[:]
    r, γeff = discountedrewards(b.rewards, b.done, learner.γ)
    advantage = r - v1.data[1]
    if γeff > 0
        h2 = learner.net(b.states[end])
        v2 = learner.valuelayer(h2)
        advantage += γeff * v2.data[1] 
    end
    Flux.back!(advantage * (-Flux.logsoftmax(p1)[b.actions[1]] + 
                            learner.αcritic * v1[1]))
    learner.opt()
end
