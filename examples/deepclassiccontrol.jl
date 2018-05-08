using TabularReinforcementLearning, Flux
include(joinpath(Pkg.dir("TabularReinforcementLearning"), "environments", 
                 "classiccontrol", "cartpole.jl"))
include(joinpath(Pkg.dir("TabularReinforcementLearning"), "environments", 
                 "classiccontrol", "mountaincar.jl"))

env = CartPole()
# env = MountainCar(maxsteps = 10^4)
ns = 4; na = 2;
learner = DQN(Chain(Dense(ns, 20, relu), Dense(20, na)),
              updateevery = 1, updatetargetevery = 100,
              doubledqn = true, replaysize = 10^3) 
learner = DeepActorCritic(Chain(Dense(ns, 20, relu)),
                          na = na, nh = 20, αcritic = 0.,
                          nsteps = 25, nenvs = 4)
# learner = ActorCriticPolicyGradient(na = na, ns = ns, αcritic = 0., nsteps = 25)
x = RLSetup(Agent(learner, 
                  callback = Progress()),
            [CartPole() for _ in 1:4],
            EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(10^5))
@time learn!(x)
