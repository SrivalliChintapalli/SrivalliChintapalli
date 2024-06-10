module mdp
using Random
using Gym
using Base

import Base: reset, step

struct Discrete
    n::Int
end

struct Box
    low::Vector{Float64}
    high::Vector{Float64}
    dtype::DataType
end

abstract type Env end

struct MarkovDecisionProcess
    trans_probs::Array{Float64, 3}
    rewards::Array{Float64, 3}
    initial_state_p::Array{Float64, 1}
    observations::Array{Float64, 2}
    feature_names::Array{String, 1}
    action_names::Array{String, 1}
    n_states_::Int
    n_actions_::Int
    n_features_in_::Int

    function MarkovDecisionProcess(trans_probs::Array{Float64, 3}, rewards::Array{Float64, 3}, initial_state_p::Array{Float64, 1}, observations::Array{Float64, 2}, feature_names::Array{String, 1}=nothing, action_names::Array{String, 1}=nothing)
        n_states_ = size(trans_probs, 1)
        n_actions_ = size(trans_probs, 3)
        n_features_in_ = size(observations, 2)

        if isnothing(feature_names)
            feature_names = ["x[$i]" for i in 1:n_features_in_]
        end

        if isnothing(action_names)
            action_names = ["action $i" for i in 1:n_actions_]
        end

        new(trans_probs, rewards, initial_state_p, observations, feature_names, action_names, n_states_, n_actions_, n_features_in_)
    end

    function remove_unreachable_states!(mdp::MarkovDecisionProcess)
        visited = Set{Int}()
        stack = findall(!iszero, mdp.initial_state_p)
        state_state_probs = sum(mdp.trans_probs, dims=3)
        
        while !isempty(stack)
            state = pop!(stack)
            push!(visited, state)
            for next_state in findall(!iszero, state_state_probs[state, :])
                if !(next_state in visited)
                    push!(stack, next_state)
                end
            end
        end
        
        if length(visited) == mdp.n_states_
            return
        end

        println("Removed states:", mdp.n_states_ - length(visited))

        visited = collect(visited)
        mdp.trans_probs = mdp.trans_probs[visited, visited, :]
        mdp.rewards = mdp.rewards[visited, visited, :]
        mdp.initial_state_p = mdp.initial_state_p[visited]
        mdp.observations = mdp.observations[visited, :]

        mdp.n_states_ = size(mdp.trans_probs, 1)
        mdp.n_actions_ = size(mdp.trans_probs, 3)
        
        nothing
    end
end

function save_mdp(mdp::MarkovDecisionProcess, filename::String)
    open(filename, "w") do file
        serialize(file, mdp)
    end
end

function optimal_return(mdp::MarkovDecisionProcess, gamma::Float64, max_iter::Int=1000000000, minimize::Bool=false)
    V = value_iteration(mdp, gamma, max_iter, minimize)
    return dot(mdp.initial_state_p, V)
end

function random_return(mdp::MarkovDecisionProcess, gamma::Float64, max_iter::Int=1000000000)
    states = 1:mdp.n_states_
    delta = 1e-10  # Error tolerance
    V = zeros(length(states))  # Initialize values with zeroes

    fixed_trans_probs = mean(mdp.trans_probs, dims=3)[:, :, 1]
    fixed_rewards = mean(mdp.rewards, dims=3)[:, :, 1]

    for _ in 1:max_iter
        V_new = sum(fixed_trans_probs .* (fixed_rewards .+ gamma .* V'), dims=2)'

        max_diff = maximum(abs.(V_new .- V))

        V = V_new

        if max_diff < delta
            break
        end
    end

    return dot(V, mdp.initial_state_p)
end

function value_iteration(mdp::MarkovDecisionProcess, gamma::Float64, max_iter::Int=1000000000, minimize::Bool=false)
    states = 1:mdp.n_states_
    delta = 1e-10
    V = zeros(length(states))

    for _ in 1:max_iter
        state_action_values = sum(mdp.trans_probs .* (mdp.rewards .+ gamma .* V'), dims=2)'

        if minimize
            best_actions = argmin(state_action_values, dims=2)
        else
            best_actions = argmax(state_action_values, dims=2)
        end

        V_new = state_action_values[CartesianIndex.(states, best_actions)]

        max_diff = maximum(abs.(V_new .- V))

        V = V_new

        if max_diff < delta
            break
        end
    end

    return V
end

function evaluate_policy(mdp::MarkovDecisionProcess, policy::Function, gamma::Float64, max_iter::Int)
    states = 1:mdp.n_states_
    delta = 1e-10
    V = zeros(length(states))

    fixed_trans_probs = similar(mdp.trans_probs[:, :, 1])
    fixed_rewards = similar(mdp.rewards[:, :, 1])

    for i in 1:mdp.n_states_
        action = policy(mdp.observations[i, :])
        fixed_trans_probs[i, :] .= mdp.trans_probs[i, :, action]
        fixed_rewards[i, :] .= mdp.rewards[i, :, action]
    end

    for _ in 1:max_iter
        V_new = sum(fixed_trans_probs .* (fixed_rewards .+ gamma .* V'), dims=2)'

        max_diff = maximum(abs.(V_new .- V))

        V = V_new

        if max_diff < delta
            break
        end
    end

    return dot(V, mdp.initial_state_p)
end

mutable struct MarkovDecisionProcessEnv <: Env
    mdp::MarkovDecisionProcess
    step_limit::Int
    random_seed::Int
    state_::Int
    step_::Int
    done::Bool
    action_space::Discrete
    observation_space::Box

    function MarkovDecisionProcessEnv(mdp::MarkovDecisionProcess, step_limit::Int=1000, random_seed::Int=123)
        action_space = Discrete(mdp.n_actions_)
        obs_low = vec(minimum(mdp.observations, dims=1))
        obs_high = vec(maximum(mdp.observations, dims=1))
        observation_space = Box(obs_low, obs_high, eltype(mdp.observations))

        new(mdp, step_limit, random_seed, 0, 0, false, action_space, observation_space)
    end
end

function custom_sample(weights::Vector{Float64})
    r = rand()
    cumulative_weight = 0.0
    for (i, weight) in enumerate(weights)
        cumulative_weight += weight
        if r < cumulative_weight
            return i
        end
    end
    return length(weights)  # Return the last index if not selected earlier
end

function reset(env::MarkovDecisionProcessEnv)
    env.state_ = custom_sample(env.mdp.initial_state_p)
    env.step_ = 0
    env.done = false

    return env.mdp.observations[env.state_, :]
end

function step(env::MarkovDecisionProcessEnv, action::Int)
    env.step_ += 1

    trans_probs = env.mdp.trans_probs[env.state_, :, action]
    next_state = custom_sample(trans_probs)

    reward = env.mdp.rewards[env.state_, next_state, action]
    observation = env.mdp.observations[next_state, :]

    env.state_ = next_state

    if env.done
        nothing
    elseif all(env.mdp.trans_probs[next_state, next_state, :] .== 1) && all(env.mdp.rewards[next_state, next_state, :] .== 0)
        env.done = true
    elseif env.step_ == env.step_limit
        env.done = true
    end

    return observation, reward, env.done, Dict()
end

function render(env::MarkovDecisionProcessEnv, mode::String="human", close::Bool=false)
    println("Current state: ", env.state_)
    println("Step: ", env.step_)
    println("Done: ", env.done)
end
    
end
