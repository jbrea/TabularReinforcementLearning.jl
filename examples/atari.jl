using TabularReinforcementLearning, Flux
# using CuArrays
loadenvironment("atari")
env = AtariEnv("pong")
na = length(getMinimalActionSet(env.ale))
model = Chain(Linear(Float64(1/255)), Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Dense(3456, 512, relu));
modeldqn = deepcopy(model);
push!(modeldqn, Dense(512, na));
learner = DQN(modeldqn, opttype = x -> Flux.RMSProp(x, .00025),
              updatetargetevery = 10^4, replaysize = 10^6, nmarkov = 4,
              startlearningat = 50000);
# learner = DeepActorCritic(model, nh = 512, policylayer = Dense(512, na),
#                           valuelayer = Dense(512, 1), 
#                           nmarkov = 4, nsteps = 5)
x = RLSetup(learner, 
            env,
            ConstantNumberSteps(100),
            preprocessor = AtariPreprocessor(gpu=false),
            callbacks = [#Progress(10^3), 
                         EvaluationPerEpisode(TotalReward())
                          ,LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)
                            ]);
# xs = toasync(x, () -> AtariEnv("../examples/atarirom_files/pong.bin"), 4)
@time learn!(x)

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
