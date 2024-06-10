import Pkg
Pkg.add("Gurobi")
Pkg.add("TextWrap")
Pkg.add("NPZ")
Pkg.add("JuMP")

#import Gurobi
#import TextWrap
#import Random
#import Printf
#import NPZ
#import JuMP

using Gurobi
using TextWrap
using Random
using Printf
using NPZ
using JuMP

include("tree.jl")
include("omdt_MDP.jl")

using .tree: Tree, TreeLeaf, TreeNode
using .mdp: MarkovDecisionProcess, evaluate_policy, value_iteration, reset, step,render

mutable struct OmdtSolver
    depth::Int
    gamma::Float64
    max_iter::Int
    delta::Float64
    n_cpus::Int
    verbose::Bool
    time_limit::Union{Nothing, Float64}
    output_dir::String
    record_progress::Bool
    seed::Int
    fixed_depth::Bool
    only_build_milp::Bool
    time_limit_::Union{Nothing, Float64}
    optimal_::Bool
    thresholds_::Union{Nothing, Vector{Vector{Float64}}}
    model_::Union{Nothing, JuMP.Model}
    recorded_objectives_::Union{Nothing, Vector{Vector{Float64}}}
    recorded_bounds_::Union{Nothing, Vector{Vector{Float64}}}
    recorded_runtimes_::Union{Nothing, Vector{Vector{Float64}}}
    optimal_values::Union{Nothing, Vector{Float64}}
    min_values::Union{Nothing, Vector{Float64}}
    trees_::Union{Nothing, Vector{Tree}}
    trees_optimal_::Union{Nothing, Vector{Bool}}
    result_summary_::Union{Nothing, Vector{Dict{String, Any}}}
    split_features_::Union{Nothing, Vector{Int}}
    split_thresholds_::Union{Nothing, Vector{Float64}}
    leaf_actions_::Union{Nothing, Vector{Int}}
end

function OmdtSolver(; depth::Int, gamma::Float64, max_iter::Int, delta::Float64, n_cpus::Int, verbose::Bool,
                     time_limit::Union{Nothing, Float64}=nothing, output_dir::String="", record_progress::Bool=false,
                     seed::Int=0, fixed_depth::Bool=false, only_build_milp::Bool=false, time_limit_::Union{Nothing, Float64}=nothing,
                     optimal_::Bool=true, thresholds_::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                     model_::Union{Nothing, JuMP.Model}=nothing, recorded_objectives_::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                     recorded_bounds_::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                     recorded_runtimes_::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                     optimal_values::Union{Nothing, Vector{Float64}}=nothing,
                     min_values::Union{Nothing, Vector{Float64}}=nothing,
                     trees_::Union{Nothing, Vector{Tree}}=nothing,
                     trees_optimal_::Union{Nothing, Vector{Bool}}=nothing,
                     result_summary_::Union{Nothing, Vector{Dict{String, Any}}}=nothing,
                     split_features_::Union{Nothing, Vector{Int}}=nothing,
                     split_thresholds_::Union{Nothing, Vector{Float64}}=nothing,
                     leaf_actions_::Union{Nothing, Vector{Int}}=nothing)
    return OmdtSolver(depth, gamma, max_iter, delta, n_cpus, verbose, time_limit, output_dir, record_progress, seed,
                      fixed_depth, only_build_milp, time_limit_, optimal_, thresholds_, model_, recorded_objectives_,
                      recorded_bounds_, recorded_runtimes_, optimal_values, min_values, trees_, trees_optimal_,
                      result_summary_, split_features_, split_thresholds_, leaf_actions_)
end


function __ancestors(node::Int)
    A_l = Int[]
    A_r = Int[]
    while node > 1
        if node % 2 == 0
            push!(A_l, node ÷ 2)
        else
            push!(A_r, node ÷ 2)
        end
        node ÷= 2
    end
    return A_l, A_r
end

function __solve_depth(
    solver::OmdtSolver,
    depth::Int,
    mdp::MarkovDecisionProcess,
    old_model::Union{Nothing, JuMP.Model}
)
    states = collect(0:mdp.n_states_-1)
    actions = collect(0:mdp.n_actions_-1)
    nodes = collect(1:2^depth-1)
    leaves = collect(2^depth:2^(depth+1)-1)

    solver.thresholds_ = [sort(unique(mdp.observations[:, j])) for j in 1:size(mdp.observations, 2)]

    all_thresholds = []
    for feature_i in 1:size(mdp.observations, 2)
        for threshold in solver.thresholds_[feature_i]
            push!(all_thresholds, (feature_i, threshold))
        end
    end

    solver.model_ = JuMP.Model(Gurobi.Optimizer)

    upper_bound = 1 / (1 - solver.gamma)
    @variable(solver.model_, pi[states, actions] >= 0)
    @constraint(solver.model_, pi .<= upper_bound)
    @variable(solver.model_, threshold[nodes, 1:length(all_thresholds)], Bin)
    @variable(solver.model_, pred_action[leaves, actions], Bin)
    @variable(solver.model_, path[1:size(mdp.observations, 1), nodes], Bin)
    @variable(solver.model_, takes_action[states, actions], Bin)

    for node in nodes
        @constraint(solver.model_, sum(threshold[node, i] for i in 1:length(all_thresholds)) == 1)
    end

    for leaf in leaves
        @constraint(solver.model_, sum(pred_action[leaf, action] for action in actions) == 1)
    end

    M = 1 / (1 - solver.gamma)

    for (i, state_observation) in enumerate(zip(states, eachrow(mdp.observations)))
        state, observation = state_observation
        println("Debug: Processing state $state with observation $observation")
        threshold_coefficients = [(observation[feat] <= thres) ? 0 : 1 for (feat, thres) in all_thresholds]
        println("Debug: threshold_coefficients for state $state: $threshold_coefficients")
        for node in nodes
            println("Debug: Creating constraint for node $node with coefficients $threshold_coefficients and thresholds $(threshold[node, :])")
            println("Debug: Dimension 1, $(axes(threshold, 1))")
            println("Debug: Dimension 2, $(axes(threshold, 2))")
            try
                @constraint(solver.model_, sum(coef * threshold[node, threshold_i] for (threshold_i, coef) in enumerate(threshold_coefficients)) == path[i+1, node])
            catch e
                println("Debug: Error at node $node, threshold_coefficients $threshold_coefficients, state $state")
                throw(e)
            end
        end
        @constraint(solver.model_, sum(takes_action[state, action] for action in actions) == 1)
        for leaf in leaves
            A_l, A_r = __ancestors(leaf)
            for action in actions
                @constraint(solver.model_, sum(1 - path[i+1, node] for node in A_l) + sum(path[i+1, node] for node in A_r) + pred_action[leaf, action] - length(A_l) - length(A_r) <= takes_action[state, action])
            end
        end
        for action in actions
            @constraint(solver.model_, pi[state, action] <= M * takes_action[state, action])
        end
    end

    for state in states
        nonzero_indices = findall(x -> x != 0, mdp.trans_probs[:, state, :])
        println("Debug: nonzero_indices for state $state: $nonzero_indices")
        @constraint(solver.model_, sum(pi[state, action] for action in actions) - sum(solver.gamma * mdp.trans_probs[other_state+1, state+1, action] * pi[other_state+1, action] for (other_state, action) in nonzero_indices) == mdp.initial_state_p[state+1])
    end

    @objective(solver.model_, Max, sum(pi[s+1, a+1] * sum(mdp.trans_probs[s+1, :, a] .* mdp.rewards[s+1, :, a]) for s in states, a in actions))

    set_optimizer_attribute(solver.model_, "OutputFlag", solver.verbose)
    set_optimizer_attribute(solver.model_, "Seed", solver.seed)
    set_optimizer_attribute(solver.model_, "Threads", solver.n_cpus)

    if !isnothing(solver.time_limit)
        set_optimizer_attribute(solver.model_, "TimeLimit", solver.time_limit_)
    end

    if !isnothing(old_model) && old_model.SolCount > 0
        for node in nodes
            for threshold_i in 1:length(all_thresholds)-1
                set_start_value(threshold[node, threshold_i], 0)
            end
            set_start_value(threshold[node, length(all_thresholds)], 1)
        end

        for var in all_variables(old_model)
            varname = JuMP.name(var)
            if occursin("pred_action", varname)
                leaf_i, action_i = parse.(Int, split(split(split(varname, "[")[2], "]")[1], ","))
                leaf_i *= 2
                set_start_value(JuMP.variable_by_name(solver.model_, "pred_action[$leaf_i,$action_i]"), JuMP.value(var))
            elseif occursin("takes_action", varname)
                state_i, action_i = parse.(Int, split(split(split(varname, "[")[2], "]")[1], ","))
                set_start_value(JuMP.variable_by_name(solver.model_, "takes_action[$state_i,$action_i]"), JuMP.value(var))
            else
                set_start_value(JuMP.variable_by_name(solver.model_, varname), JuMP.value(var))
            end
        end
    end

    if solver.only_build_milp
        return
    end

    if solver.record_progress
        objectives = Float64[]
        bounds = Float64[]
        runtimes = Float64[]

        function callback(cb)
            where = callback_where(cb)
            if where == JuMP.MIPNODE
                push!(objectives, callback_node_obj(cb))
                push!(bounds, callback_node_bound(cb))
                push!(runtimes, callback_runtime(cb))
            end
        end

        optimize!(solver.model_, callback=callback)

        push!(objectives, objective_value(solver.model_))
        push!(bounds, objective_bound(solver.model_))
        push!(runtimes, JuMP.objective_value(solver.model_))

        push!(solver.recorded_objectives_, objectives)
        push!(solver.recorded_bounds_, bounds)
        push!(solver.recorded_runtimes_, runtimes)
    else
        optimize!(solver.model_)
    end

    if has_values(solver.model_)
        __model_vars_to_tree_new(solver, mdp, nodes, leaves, threshold, all_thresholds, pred_action, depth)
    else
        solver.tree_policy_ = Tree(TreeLeaf(0))
        solver.optimal_ = false
        solver.objective_ = evaluate_policy(mdp, solver.tree_policy_.act, solver.gamma, 1000000000)
        solver.bound_ = JuMP.objective_bound(solver.model_)
    end
end






function __model_vars_to_tree_new(
    solver::OmdtSolver,
    mdp::MarkovDecisionProcess,
    nodes::Array{Int, 1},
    leaves::Array{Int, 1},
    threshold::Dict,
    all_thresholds::Array{Tuple{Int, Float64}, 2},
    pred_action::Dict,
    depth::Int
)
    actions = collect(0:mdp.n_actions_-1)
    solver.split_features_ = Int[]
    solver.split_thresholds_ = Float64[]
    solver.leaf_actions_ = Int[]
    for node in nodes
        threshold_i = argmax([JuMP.value(threshold[node, i]) for i in 1:size(threshold, 2)])  # argmax along rows
        feature_i, threshold_value = all_thresholds[threshold_i]

        push!(solver.split_features_, feature_i)
        push!(solver.split_thresholds_, threshold_value)
    end

    for leaf in leaves
        leaf_action = argmax([JuMP.value(pred_action[leaf, a]) for a in actions])  # argmax along rows
        push!(solver.leaf_actions_, leaf_action)
    end

    tree_nodes = Any[]
    for (feature, threshold) in zip(solver.split_features_, solver.split_thresholds_)
        push!(tree_nodes, TreeNode(feature, threshold, nothing, nothing))
    end
    for action in solver.leaf_actions_
        push!(tree_nodes, TreeLeaf(action))
    end
    for node_i in nodes
        tree_nodes[node_i] = TreeNode(tree_nodes[node_i], tree_nodes[2 * node_i - 1], tree_nodes[2 * node_i])
    end

    solver.tree_policy_ = Tree(tree_nodes[1])
    prune!(solver.tree_policy_)

    if solver.verbose > 0
        println("Tree policy:")
        println(to_string(solver.tree_policy_, mdp.feature_names, mdp.action_names))

        println("Optimal decision tree (depth=$(int(log2(length(solver.split_thresholds_) + 1)))) value: $(objective_value(solver.model_))")
    end

    optimal = JuMP.termination_status(solver.model_) == JuMP.OPTIMAL
    optimal_value = dot(mdp.initial_state_p, solver.optimal_values)

    solver.optimal_ = optimal
    solver.objective_ = objective_value(solver.model_)
    solver.bound_ = objective_bound(solver.model_)

    push!(solver.result_summary_, Dict(
        "objective" => solver.objective_,
        "bound" => solver.bound_,
        "VI objective" => optimal_value,
        "optimal" => optimal,
        "runtime" => JuMP.runtime(solver.model_),
        "depth" => depth,
        "max_iter" => solver.max_iter,
        "delta" => solver.delta,
        "seed" => solver.seed
    ))

    open(f -> begin
        line_width = 60
        code = ""
        code *= "# Properties ".center(line_width, '#') * "\n"
        code *= "# expected discounted reward: $(objective_value(solver.model_))\n"
        code *= "# expected discounted reward bound: $(objective_bound(solver.model_))\n"
        code *= "# value iteration: $optimal_value\n"
        code *= "# proven optimal: $optimal\n"
        code *= "# runtime: $(JuMP.runtime(solver.model_))\n"
        code *= "# Parameters ".center(line_width, '#') * "\n"
        code *= "# depth: $depth\n"
        code *= "# gamma: $(solver.gamma)\n"
        code *= "# max_iter: $(solver.max_iter)\n"
        code *= "# delta: $(solver.delta)\n"
        code *= "# seed: $(solver.seed)\n"
        code *= '#' ^ line_width * "\n"

        code *= "function act($(join(mdp.feature_names, ", ")))\n"
        tree_code = to_string(solver.tree_policy_, mdp.feature_names, mdp.action_names)
        code *= "$(TextWrap.indent(tree_code, "    "))\n"

        write(f, code)
    end, "$(solver.output_dir)policy_depth_$(depth)_seed_$(solver.seed).py", "w")
end

function solve(
    solver::OmdtSolver,
    mdp::MarkovDecisionProcess
)
    solver.optimal_values = value_iteration(mdp, gamma=solver.gamma)
    solver.min_values = value_iteration(mdp, gamma=solver.gamma, minimize=true)

    solver.trees_ = Tree[]
    solver.trees_optimal_ = Bool[]
    solver.result_summary_ = Dict[]

    if solver.record_progress
        solver.recorded_objectives_ = Array{Float64}[]
        solver.recorded_bounds_ = Array{Float64}[]
        solver.recorded_runtimes_ = Array{Float64}[]
    end

    if solver.fixed_depth
        if solver.verbose
            println("Starting with fixed depth $(solver.depth)")
        end

        __solve_depth(
            solver,
            solver.depth,
            mdp,
            nothing
        )

        if solver.only_build_milp
            return
        end

        push!(solver.trees_, solver.tree_policy_)
    else
        old_model = nothing
        for depth in 1:solver.depth
            if solver.verbose
                println("Starting with depth $depth")
            end

            __solve_depth(
                solver,
                depth,
                mdp,
                old_model
            )

            push!(solver.trees_, solver.tree_policy_)
            push!(solver.trees_optimal_, solver.optimal_)

            old_model = solver.model_
        end

        for (tree, optimal) in zip(reverse(solver.trees_), reverse(solver.trees_optimal_))
            if !isnothing(tree)
                solver.tree_policy_ = tree
                solver.optimal_ = optimal
                break
            end
        end
    end
end

function act(solver::OmdtSolver, observation)
    """
    Return the next action given the observation according to the learned tree.
    """
    return act(solver.tree_policy_, observation)  # You need to define act for the Tree type separately
end

# Sample test
function main()
    # Define a simple MDP
    trans_probs = zeros(2, 2, 2)
    trans_probs[:, 1, 1] = [0.8, 0.2]
    trans_probs[:, 2, 1] = [0.1, 0.9]
    trans_probs[:, 1, 2] = [0.9, 0.1]
    trans_probs[:, 2, 2] = [0.2, 0.8]

    rewards = zeros(2, 2, 2)
    rewards[:, 1, 1] = [1.0, 0.0]
    rewards[:, 2, 1] = [0.0, 1.0]
    rewards[:, 1, 2] = [0.5, 0.5]
    rewards[:, 2, 2] = [0.5, 0.5]

    initial_state_p = [0.5, 0.5]
    observations = [0.0 0.0; 1.0 1.0]
    feature_names = ["feature1", "feature2"]
    action_names = ["action1", "action2"]

    mdp = MarkovDecisionProcess(trans_probs, rewards, initial_state_p, observations, feature_names, action_names)

    solver = OmdtSolver(
        depth=2,
        gamma=0.9,
        max_iter=100,
        delta=0.01,
        n_cpus=1,
        verbose=true,
        time_limit=nothing,
        output_dir="",
        record_progress=false,
        seed=123,
        fixed_depth=false,
        only_build_milp=false,
        time_limit_=nothing,
        optimal_=true
    )

    __solve_depth(solver, solver.depth, mdp, nothing)

    println("Solver completed successfully.")
end

main()
