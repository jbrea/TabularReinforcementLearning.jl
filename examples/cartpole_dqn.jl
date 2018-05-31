using TabularReinforcementLearning, Flux
@everywhere include(joinpath(Pkg.dir("TabularReinforcementLearning"), 
                             "environments", "classiccontrol", "cartpole.jl"))

env = CartPole();
learner = DQN(Chain(Dense(4, 16, relu), Dense(16, 16, relu), Dense(16, 2)),
              minibatchsize = 32, doubledqn = true, γ = .99, loss = huberloss,
              opttype = x -> ADAM(x, .001), updateevery = 1,
              replaysize = 1000, updatetargetevery = 100, startlearningat = 100);
x = RLSetup(learner,
            env,
            ConstantNumberSteps(10^5),
            callbacks = [EvaluationPerEpisode(TotalReward()), Progress()]);
@time learn!(x)
x.callbacks[1].values
