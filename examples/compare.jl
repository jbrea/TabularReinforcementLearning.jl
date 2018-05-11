# run this file with julia -p 4 to use 4 multiple cores in the comparison
@everywhere begin
using TabularReinforcementLearning, Flux
loadenvironment("cartpole")
getenv() = (CartPole(), 4, 2)
setup(learner, env) = RLSetup(learner, env, ConstantNumberEpisodes(8000),
                              callbacks = [EvaluateGreedy(callback =
                                        EvaluationPerEpisode(TotalReward(),
                                                            returnmean = true),
                                        stoppingcriterion =
                                        ConstantNumberEpisodes(200),
                                        every = Episode(200))])
function acpg(i)
    env, ns, na = getenv()
    learner = ActorCriticPolicyGradient(na = na, ns = ns, α = .01,
                                        αcritic = 0.01, nsteps = 25)
    setup(learner, env)
end
function dqn(i)
    env, ns, na = getenv()
    learner = DQN(Chain(Dense(ns, 20, relu), Dense(20, na)),
                  updateevery = 1, updatetargetevery = 100,
                  startlearningat = 200, minibatchsize = 16,
                  doubledqn = true, replaysize = 10^3) 
    setup(learner, env)
end
rlsetupcreators = Dict("linear ACPG" => acpg, "DQN" => dqn)
end # everywhere
@time res = compare(rlsetupcreators, 20, verbose = true)

using JLD2
@save tempname() * ".jld2" res

a = plotcomparison(res);
a["legend pos"] = "south east";
a["x label"] = "epochs";
a["y label"] = "average episode length greedy policy"
a
