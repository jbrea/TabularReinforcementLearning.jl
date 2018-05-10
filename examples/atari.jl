include(joinpath(Pkg.dir("TabularReinforcementLearning"),
                 "environments", "atari.jl"))

using TabularReinforcementLearning, Flux
# using CuArrays
env = AtariEnv("../examples/atarirom_files/pong.bin")
na = length(getMinimalActionSet(env.ale))
model = Chain(Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Dense(3136, 512, relu));
modeldqn = deepcopy(model);
push!(modeldqn, Dense(512, na));
learner = DQN(modeldqn, 
              updatetargetevery = 10^4, replaysize = 10^5, nmarkov = 4,
              startlearningat = 50000);
learner = DeepActorCritic(model, nh = 512, policylayer = Dense(512, na),
                          valuelayer = Dense(512, 1), 
                          nmarkov = 4, nsteps = 5)
x = RLSetup(learner, 
            env,
            ConstantNumberSteps(10^5),
            preprocessor = AtariPreprocessor(),
            callbacks = [Progress(10^1), EvaluationPerEpisode(TotalReward())
#                          ,LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)
                            ]);
xs = toasync(x, 16)
@time learn!(xs)

env = AtariEnv("examples/atarirom_files/pong.bin", colorspace = "Raw")
na = length(getMinimalActionSet(env.ale))
learner2 = Sarsa(na = na, ns = 20652352, λ = .9, γ = .99, α = .05, 
                 tracekind = ReplacingTraces)
x2 = RLSetup(Agent(learner2, policy = EpsilonGreedyPolicy(.01),
             preprocessor = AtariBPROST(),
             callback = Progress(10^3)),
             env,
             EvaluationPerEpisode(TotalReward()),
             ConstantNumberSteps(10^7));
@time learn!(x2)
