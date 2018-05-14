# run this file with julia -p 4 to use 4 cores in the comparison
@everywhere begin
using TabularReinforcementLearning, Flux
loadenvironment("cartpole")
getenv() = (CartPole(), 4, 2)
function setup(learner, env, preprocessor = NoPreprocessor())
    cb = EvaluateGreedy(callback = EvaluationPerEpisode(TotalReward(),
                                                        returnmean = true),
                        stoppingcriterion = ConstantNumberEpisodes(200),
                        every = Episode(200))
    RLSetup(learner, env, ConstantNumberEpisodes(2000), 
            callbacks = [cb], preprocessor = preprocessor)
end
function acpg(i)
    env, ns, na = getenv()
    learner = ActorCriticPolicyGradient(na = na, ns = ns, α = .02,
                                        αcritic = 0.01, nsteps = 25)
    setup(learner, env)
end
function dqn(i)
    env, ns, na = getenv()
    learner = DQN(Chain(Dense(ns, 48, relu), Dense(48, 24, relu), Dense(24, na)),
                  updateevery = 1, updatetargetevery = 100,
                  startlearningat = 50, minibatchsize = 32,
                  doubledqn = true, replaysize = 10^3, 
                  opttype = x -> ADAM(x, .0005)) 
    setup(learner, env)
end
function tilingsarsa(i)
    env, ns, na = getenv()
    p0 = StateAggregator([-2.6, -4, -.24, -3.4], [2.6, 4, .24, 3.4], 10 * ones(4))
    preprocessor = TilingStateAggregator(p0, 10)
    learner = Sarsa(ns = 10*10^4, na = 2)
    setup(learner, env, preprocessor)
end
rlsetupcreators = Dict("linear ACPG" => acpg, "DQN" => dqn, 
                       "tiling Sarsa" => tilingsarsa)
end # everywhere
@time res = compare(rlsetupcreators, 20, verbose = true)

using JLD2
@save tempname() * ".jld2" res

a = plotcomparison(res);
a["legend pos"] = "south east";
a["xlabel"] = "epochs";
a["ylabel"] = "average episode length greedy policy"
a
