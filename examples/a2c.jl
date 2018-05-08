using TabularReinforcementLearning
@everywhere include(joinpath(Pkg.dir("TabularReinforcementLearning"), "environments", "classiccontrol", "cartpole.jl"))
@everywhere include(joinpath(Pkg.dir("TabularReinforcementLearning"), "environments", "classiccontrol", "mountaincar.jl"))


# used for testing

# julia> learner = A2C(Chain(Dense(4, 3)),
#                      replaysize = 26, nsteps = 25, batchsize = 1, γ = .9);
# 
# julia> x = RLSetup(Agent(learner), env, EvaluationPerEpisode(TotalReward()),
#                    ConstantNumberSteps(10^5));    
# julia> @time learn!(x)  
# 126.331171 seconds (267.88 M allocations: 13.034 GiB, 2.39% gc time)
#julia> learner = A2C(Chain(Dense(4, 30), Dense(30, 3)), 
#                      replaysize = 26, nsteps = 25, batchsize = 1, γ = .9);
# julia> x = RLSetup(Agent(learner), env, EvaluationPerEpisode(TotalReward()),
#                    ConstantNumberSteps(10^3));
# julia> @time learn!(x)
#  12.227629 seconds (13.33 M allocations: 725.842 MiB, 2.28% gc time)

x = RLSetup(ActorCriticPolicyGradient(ns = 4, na = 2, αcritic = 0., nsteps = 25),
            CartPole(),
            ConstantNumberSteps(10^5),
            callbacks = [EvaluationPerEpisode(TotalReward())])
@time learn!(x)
pgfplot(Plot(Coordinates(1:length(x.callbacks[1].values), x.callbacks[1].values)), "/tmp/juliaF9BrwQ.pdf")

x = RLSetup(ActorCriticPolicyGradient(ns = 4, na = 2, αcritic = 0., 
                                      nsteps = 25, α = .1/4),
            CartPole(),
            ConstantNumberSteps(10^5),
            callbacks = [EvaluationPerEpisode(TotalReward())])
rlsetups = replicate_with_shared_params(x, 4)
@time learn!(rlsetups)
lcs = [x.callbacks[1].values for x in rlsetups]
pgfplot(@pgf(Axis({no_markers}, [PlotInc(Coordinates(1:length(lc), lc)) 
                                 for lc in lcs]...)), "/tmp/asdasd.pdf")

import TabularReinforcementLearning:Id
x = RLSetup(DeepActorCritic(model = Id(), nh = 4, na = 2, αcritic = 0., nsteps = 25),
            CartPole(),
            ConstantNumberSteps(10^5),
            callbacks = [EvaluationPerEpisode(TotalReward())])
@time learn!(x)
pgfplot(Plot(Coordinates(1:length(x.callbacks[1].values), x.callbacks[1].values)), "/tmp/juliaF9BrwQ.pdf")

envs = [CartPole() for _ in 1:4];
agents = ParallelAgents(A2C(TabularReinforcementLearning.Chain(Id()), 
               nsteps = 25, γ = .9,
               αcritic = 0., nh = 4, na = 2, opt = () -> Knet.Sgd(lr = .1)), 4);
x = RLSetup(agents, envs, EvaluationPerEpisode(TotalReward()), 
            ConstantNumberSteps(10^5));
@time learn!(x)
# pgfplot(Plot(Coordinates(1:length(x.metric.values), x.metric.values)), "/tmp/juliaF9BrwQ.pdf")
xeval = RLSetup(Agent(agents.learner[1], policy = SoftmaxPolicy1()), CartPole(), 
                EvaluationPerEpisode(TotalReward()), ConstantNumberSteps(10^4))
run!(xeval)
xeval.metric.values;
pgfplot(Plot(Coordinates(1:length(xeval.metric.values), xeval.metric.values)), "/tmp/juliaF9BrwQ.pdf")

# pgfplot(Plot(Coordinates(1:length(x0.metric.values), x0.metric.values)), "/tmp/julia0F9BrwQ.pdf")

srand(81020891);
w0 = rand(2, 4);
env = ForcedEpisode([rand(4) for _ in 1:121], [i == 16 for i in 1:121], rand(20))
pol = ForcedPolicy(rand(1:2, 121))
using Flux
learner = A2C2(TabularReinforcementLearning.Chain(Id()), nh = 4, na = 2, nsteps = 3,
               αcritic = 0., opt = () -> Knet.Sgd(lr = .1))
# learner.policylayer.W.data[:] = w0
x = RLSetup(Agent(learner, policy = pol), env, TotalReward(),
            ConstantNumberSteps(90))
learn!(x)
reset!(env); pol.t = 1
learner2 = ActorCriticPolicyGradient(ns = 4, na = 2, nsteps = 3, 
                                     αcritic = 0.);
# learner2.params[:] = w0
import TabularReinforcementLearning.selectaction
selectaction(::PolicyGradientForward, p::ForcedPolicy, s) =  selectaction(p, s)
x2 = RLSetup(Agent(learner2, policy = pol), env, TotalReward(),
             ConstantNumberSteps(90));
learn!(x2)
learner.model.policylayer.w
learner2.params


env = MountainCar(maxsteps = 10^4)
# using Flux
nenv = 8
envs = [CartPole() for _ in 1:nenv];
# learner = DeepActorCritic(Flux.Chain(Id()), nh = 4, na = 3, nsteps = 25,
#                           αcritic = 0., opt = p -> Flux.SGD(p, .1));
learner = ActorCriticPolicyGradient(na = 3, ns = 4, α = .03/nenv, αcritic = 0., nsteps = 25)
x = RLSetup(AsyncParallelAgents(learner, nenv), envs, EvaluationPerEpisode(TotalReward()), 
            ConstantNumberSteps(10^6));
# x = RLSetup(Agent(learner), envs[1], EvaluationPerEpisode(TotalReward()), 
#             ConstantNumberSteps(10^5));
@time learn!(x)
pgfplot(Plot(Coordinates(1:length(x.metric.values), x.metric.values)), "/tmp/julia0F9BrwQ.pdf")

xeval = RLSetup(Agent(learner), env, 
                EvaluationPerEpisode(TotalReward()), ConstantNumberSteps(10^4))
run!(xeval)
xeval.metric.values
pgfplot(Plot(Coordinates(1:length(xeval.metric.values), xeval.metric.values)), "/tmp/juliaF9BrwQ.pdf")


env = CartPole();
using Flux
learner = DQN(Chain(Dense(4, 16, relu), Dense(16, 16, relu),
                    Dense(16, 2)),
              minibatchsize = 32, doubledqn = true, γ = .99, 
              opttype = x -> ADAM(x, .001), updateevery = 1,
              replaysize = 1000, updatetargetevery = 100, startlearningat = 100);
# learner = DeepActorCritic(Chain(Dense(4, 20, relu)), nenvs = 10,
#                           statetype = Int64,
#                            nh = 20, na = 2, nsteps = 25, αcritic = 0.);
# # # learner = ActorCriticPolicyGradient(ns = 4, na = 2, nsteps = 25, αcritic = 0.);
x = RLSetup(learner
#                    , callback = LinearDecreaseEpsilon(1, 4*10^4, .5, .01)
                 , 
#               [CartPole() for _ in 1:10], 
               env,
               ConstantNumberSteps(10^5),
               callbacks = [EvaluationPerEpisode(TotalReward())],
#             RecordAll(),
                  );
@time learn!(x)
pgfplot(Plot(Coordinates(1:length(x.callbacks[1].values), x.callbacks[1].values)), "/tmp/juliaF9BrwQ.pdf")

w = SharedArray(zeros(2, 4))
ls = [DeepActorCritic(Id(), policylayer = Linear(w)) for _ in 1:2];
envs = [CartPole() for _ in 1:2];
x = RLSetup(ParallelAgents(ls), envs, EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(10^4));
@time learn!(x)

learner2 = ActorCriticPolicyGradient(ns = 4, na = 2, nsteps = 25, 
                                     αcritic = 0.);
x2 = RLSetup(Agent(learner2), env, EvaluationPerEpisode(TotalReward()),
             ConstantNumberSteps(10^5));
reset!(env)
@time learn!(x2)
pgfplot(Plot(Coordinates(1:length(x2.metric.values), x2.metric.values)), "/tmp/juliaF9BrwQ.pdf")

xeval = RLSetup(Agent(learner2), CartPole(), 
                EvaluationPerEpisode(TotalReward()), ConstantNumberSteps(10^4))
run!(xeval)
xeval.metric.values
pgfplot(Plot(Coordinates(1:length(xeval.metric.values), xeval.metric.values)), "/tmp/juliaF9BrwQ.pdf")

envs = [CartPole() for _ in 1:4];
agents = ParallelAgents(A2C(Chain(Id()), nh = 4, na = 2, nsteps = 25, 
#                                                   α = .1/4,
opt = (w, g) -> Knet.Sgd(w, g; lr = .1/4),
                                                  αcritic = 0.), 4);
x = RLSetup(agents, envs, EvaluationPerEpisode(TotalReward()),
            ConstantNumberSteps(10^5));
@time learn!(x)
xeval = RLSetup(Agent(agents.learner[1], policy = SoftmaxPolicy1()), CartPole(), 
                EvaluationPerEpisode(TotalReward()), ConstantNumberSteps(10^4))
run!(xeval)
xeval.metric.values
pgfplot(Plot(Coordinates(1:length(xeval.metric.values), xeval.metric.values)), "/tmp/juliaF9BrwQ.pdf")

env = MountainCar(maxsteps = 10^5)
learner3 = Sarsa(ns = 50^2, na = 3, λ = 0., nsteps = 25, discretestates = true);
x3 = RLSetup(Agent(learner3, 
#                     preprocessor = SparseRandomProjection(randn(225,
#                                                                 2)/sqrt(225),
#                                                           zeros(225))
#                    preprocessor = RadialBasisFunctions(env.observation_space,
#                                                        225, .1)
                   preprocessor = StateAggregator([-1.2, -.07], [.6, .07], 
                                                  fill(50, 2), perdimension = false)
                  ), 
             env, EvaluationPerEpisode(TimeSteps()),
             ConstantNumberSteps(10^6));
reset!(env)
@time learn!(x3)
pgfplot(Plot(Coordinates(1:length(x3.metric.values), x3.metric.values)), "/tmp/juliaF9BrwQ.pdf")
@pgf pgfplot(Axis({ymax = 400}, Plot(Coordinates(1:length(x3.metric.values), x3.metric.values))), "/tmp/juliaF9BrwQ.pdf")

T = TabularReinforcementLearning
learner = A2C(Chain(Dense(4, 3, initfun = zeros)), 
              replaysize = 2, nsteps = 1, αcritic = 0.)
T.pushstateaction!(learner.buffer, [1., 0., 0., 0.], 1)
T.pushstateaction!(learner.buffer, [0., 1., 0., 0.], 3)
T.pushreturn!(learner.buffer, 2., false)
buffer = learner.buffer
indices = [1]
xstart = hcat(buffer.states[indices]...)
xend = hcat(buffer.states[indices+buffer.nsteps]...)
ystart = learner.model(xstart)
yend = learner.model(xend)
g = T.actorcriticgradfun(learner.model.w, learner.model.chain, learner.αcritic,
                         xstart,
                         xend,
                         buffer.actions[indices],
                         hcat(buffer.rewardsums[indices]...))

learner2 = ActorCriticPolicyGradient(ns = 4, na = 2, nsteps = 1)
T.pushstateaction!(learner2.buffer, [1., 0., 0., 0.], 1)
T.pushstateaction!(learner2.buffer, [0., 1., 0., 0.], 3)
T.pushreturn!(learner2.buffer, 2., false)
update!(learner2)
learner2.params

@everywhere begin
struct A
    w
end
struct B
    x
end
end
w = SharedArray(zeros(3, 2))
as = [A(w) for _ in 1:3]
bs = [B(i) for i in 1:3]
@sync @parallel for i in 1:3
    as[i].w .+= bs[i].x
end

c = Flux.Chain(Linear(4, 2))
w = rand(2, 4); wc = deepcopy(w); c.layers[1].W.data[:] = wc
x = rand(4)
y = c(x)
Flux.back!(Flux.logsoftmax(y)[1])
g = deepcopy(c.layers[1].W.grad)
p = exp.(w * x)/sum(exp.(w * x))
g2 = -p * x'
g2[1, :] .+= x
g2 ≈ g
