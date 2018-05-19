using TabularReinforcementLearning, Flux
const withgpu = false
if withgpu 
    using CuArrays
    const inputdtype = Float32
else
    const inputdtype = Float64
end
loadenvironment("atari")
env = AtariEnv("pong")
na = length(getMinimalActionSet(env.ale))
model = Chain(x -> inputdtype.(x./255), Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
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
            ConstantNumberSteps(5*10^5),
            preprocessor = AtariPreprocessor(gpu=withgpu),
            callbacks = [Progress(5*10^3), 
                         EvaluationPerEpisode(TotalReward()),
                         LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)]);
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

env = AtariEnv("breakout")
na = length(getMinimalActionSet(env.ale))
model = Chain(x -> Float64.(x./255), Conv((8, 8), 4 => 16, relu,
                                          stride = (4, 4), pad = (4, 4)), 
                         Conv((4, 4), 16 => 32, relu, 
                              stride = (2, 2), pad = (2, 2)),
                         x -> reshape(x, :, size(x, 4)),
                         Dense(3872, 512, relu));
modeldqn = deepcopy(model);
push!(modeldqn, Dense(512, na));
learner = DQN(modeldqn, opttype = x -> Flux.RMSProp(x, .00025),
              updatetargetevery = 10^4, replaysize = 10^6, nmarkov = 4,
              startlearningat = 50000);
x = RLSetup(learner, 
            env,
            ConstantNumberSteps(10^7),
            preprocessor = AtariPreprocessor(gpu=false, croptosquare = true),
            callbacks = [Progress(10^3), 
                         EvaluationPerEpisode(TotalReward()),
                         LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)]);
@time learn!(x)

