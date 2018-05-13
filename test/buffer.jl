import TabularReinforcementLearning: ArrayCircularBuffer, ArrayStateBuffer, 
pushstateaction!, pushreturn!

a = ArrayCircularBuffer(Array, Int64, (1), 8)
for i in 1:5 push!(a, i) end
@test a[1] == [1]
@test a[endof(a)] == [5]
@test endof(a) == 5
@test getindex(a, 5, 3)[:] == collect(3:5)
@test getindex(a, endof(a), 4)[:] == collect(2:5)
for i in 6:12 push!(a, i) end
@test a[1] == [5]
@test a[endof(a)] == [12]
@test getindex(a, endof(a), 4)[:] == collect(9:12)

a = ArrayStateBuffer(capacity = 7)
for i in 1:4
    pushstateaction!(a, [i], i)
    pushreturn!(a, i, false)
end
pushstateaction!(a, [5], 5)
@test a.states[3][1] == a.actions[3] == a.rewards[3]
pushreturn!(a, 5, false)
for i in 6:9
    pushstateaction!(a, [i], i)
    pushreturn!(a, i, false)
end
pushstateaction!(a, [10], 10)
@test a.states[3][1] == a.actions[3] == a.rewards[3]

for T in [7, 97]
    p = ForcedPolicy(rand(1:4, 100))
    ends = rand(1:100, 10)
    env = ForcedEpisode(rand(1:10, 100), [i in ends ? true : false for i in 1:100], rand(100))
    x = RLSetup(1, env, ConstantNumberSteps(98), policy = p, 
                buffer = ArrayStateBuffer(capacity = 10, datatype = Int64), 
                callbacks = [RecordAll()],
                fillbuffer = true, islearning = false)
    learn!(x)
    @test x.buffer.actions[end-5:end] == x.callbacks[1].actions[end-5:end]
    @test x.buffer.done[end-5:end] == x.callbacks[1].done[end-5:end]
    @test x.buffer.states[end-5:end][:] == Int64.(x.callbacks[1].states[end-5:end])
    @test x.buffer.rewards[end-4:end] == x.callbacks[1].rewards[end-4:end]
end
