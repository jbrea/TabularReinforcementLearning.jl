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
model = Chain(x -> x./inputdtype(255), Conv((8, 8), 4 => 32, relu, stride = (4, 4)), 
                         Conv((4, 4), 32 => 64, relu, stride = (2, 2)),
                         Conv((3, 3), 64 => 64, relu),
                         x -> reshape(x, :, size(x, 4)),
                         Dense(3456, 512, relu), 
                         Dense(512, na));
learner = DQN(model, opttype = x -> Flux.ADAM(x, .0001), loss = huberloss(),
              updatetargetevery = 10^4, replaysize = 10^6, nmarkov = 4,
              startlearningat = 50000);
x = RLSetup(learner, 
            env,
            ConstantNumberSteps(5*10^7),
            preprocessor = AtariPreprocessor(gpu=withgpu),
            callbacks = [Progress(5*10^2), 
                         EvaluationPerEpisode(TotalReward()),
                         LinearDecreaseEpsilon(5 * 10^4, 10^6, 1, .1)]);
@time learn!(x)
