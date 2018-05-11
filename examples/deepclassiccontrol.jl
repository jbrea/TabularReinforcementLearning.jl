using TabularReinforcementLearning, Flux
loadenvironment("cartpole")

env = CartPole()
# env = MountainCar(maxsteps = 10^4)
ns = 4; na = 2;
learner = DQN(Chain(Dense(ns, 20, relu), Dense(20, na)),
              updateevery = 1, updatetargetevery = 100,
              startlearningat = 200, minibatchsize = 16,
              doubledqn = true, replaysize = 10^3) 
# learner = DeepActorCritic(Chain(Dense(ns, 20, relu)),
#                           na = na, nh = 20, αcritic = 0.,
#                           nsteps = 25)
learner = ActorCriticPolicyGradient(na = na, ns = ns, αcritic = 0.1, nsteps = 25)
struct FP end
import TabularReinforcementLearning:preprocessstate
preprocessstate(::FP, s) = Float32.(s)
x = RLSetup(learner,
            env,
            ConstantNumberSteps(2*10^5),
#             preprocessor = FP(),
            callbacks = [EvaluationPerEpisode(TotalReward()), Progress(),
                         SaveLearner(every = Step(2*10^4))])
@time learn!(x)


a = @pgf Axis({no_markers}, Plot(Coordinates(collect(1:length(x.callbacks[1].values)),
                                             x.callbacks[1].values)),
              PlotInc(Coordinates(collect(1:length(x.callbacks[3].callbacks[1].values)), x.callbacks[3].callbacks[1].values)));
pgfplot(a, "/tmp/juliaYMdQXc.pdf")
