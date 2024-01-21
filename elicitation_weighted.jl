using Random
using LinearAlgebra
using StatsBase
using JuMP
using COPT
using Combinatorics

include("MOKP.jl")


"""
Generate random weight vector of length n normalized the sum to 1.
"""
function random_weight(n::Int)
    return normalize(rand(n))
end


"""
Return a boolean indicating the preference between two solutions x and y according to the weighted sum
"""
function is_preferred(weight::Vector{Float64}, x::Vector{Int}, y::Vector{Int})
    return wsum(x, weight) > wsum(y, weight)
end

"""Given a dict of PMR (pairwise max regrets), return a tuple (y, (regret, model)) where y is the solution that could make the decision-maker regret the most if they choose x, 'regret' is the amount of regret for choosing x instead of y, and the model (Linear Program) that was used to calculate this regret.
"""
function max_regret(PMR::Dict{Pareto,Dict{Pareto,Tuple{Float64,Model}}}, x::Pareto)
    mr = nothing
    for (y, value) in PMR[x]
        regret, model = value
        if isnothing(mr)
            mr = (y, (regret, model))
        elseif mr[2][1] < regret
            mr = (y, (regret, model))
        end
    end
    return mr
end


"""
Allows for the incremental elicitation of the preferences of a randomly chosen decision-maker whose preferences are represented by a weighted sum. It is assumed that we know a number of the decision-maker's preferences.
"""
function incremental_elicitation(p::Int, X::Vector{Pareto}, number_of_known_preferences::Int; MMRlimit::Float64=0.001, weight::Union{Vector{Float64},Nothing}=nothing)
    if isnothing(weight)
        weight = random_weight(p)
    end
    # println("weight: ", weight)
    if length(X) == 1
        return X[1], 0, wsum(X[1].objectives, weight), weight
    end
    # println("number_of_known_preferences: ", number_of_known_preferences)
    all_pairs_of_solutions = collect(combinations(X, 2))
    # random choice number_of_known_preferences pairs of solutions
    known_preferences = sample(all_pairs_of_solutions, number_of_known_preferences, replace=false)
    preferences = Tuple{Pareto,Pareto}[]
    for (x, y) in known_preferences
        if is_preferred(weight, x.objectives, y.objectives)
            push!(preferences, (x, y))
        elseif is_preferred(weight, y.objectives, x.objectives)
            push!(preferences, (y, x))
        end
    end

    println("itération n°1")
    MMR = one_question_elicitation(X, preferences, weight)

    if isnothing(MMR)
        return X[1], 0, wsum(X[1].objectives, weight), weight
    end

    println("Question : \nx : $(MMR[1].objectives)\ny : $(MMR[2][1].objectives)\nregret : $(MMR[2][2][1])")
    num_question = 1

    while (MMR[2][2][1] > MMRlimit)
        num_question += 1
        println("iteration: ", num_question)
        MMR = one_question_elicitation(X, preferences, weight)
        println("Question : \nx : $(MMR[1].objectives)\ny : $(MMR[2][1].objectives)\nregret : $(MMR[2][2][1])")
    end

    value_optimal = wsum(MMR[1].objectives, weight)

    return MMR[1], num_question, value_optimal, weight
end

"""
- Calculates the PMR (Pairwise Max Regret)
- Deduces the MR (Max Regret)
- Deduces the MMR (Maximum Max Regret)
- Deduces the question, asks it, and subsequently adjusts the weight space accordingly

finding the best question that most significantly reduces the weight space and then asking it.

# Parameters
- `X::Vector`: A list of all potentially Pareto optimal solutions found with PLS.
- `preferences::Array`: A list of pairs indicating the decision-maker's preferences between solutions in X.
- `weight::Vector`: A list of weights for the decision-maker.
"""
function one_question_elicitation(X::Vector{Pareto}, preferences::Vector{Tuple{Pareto,Pareto}}, weight::Vector{Float64})
    p = length(weight)
    PMR = Dict{Pareto,Dict{Pareto,Tuple{Float64,Model}}}()
    for x in X
        PMR[x] = Dict{Pareto,Tuple{Float64,Model}}()
    end

    for x in X
        for y in X
            if isequal(x, y)
                continue
            end
            model = JuMP.Model(COPT.Optimizer)
            set_silent(model)
            @variable(model, w[1:p])
            @objective(model, Max, (y.objectives' * w) - (x.objectives' * w))

            for (x_pref, y_pref) in preferences
                @constraint(model, (x_pref.objectives' * w) >= (y_pref.objectives' * w))
            end

            for i in 1:p
                t = zeros(p)
                t[i] = 1
                @constraint(model, t' * w <= 1)
                @constraint(model, t' * w >= 0)
            end
            @constraint(model, sum(w) == 1)

            optimize!(model)
            if termination_status(model) == MOI.OPTIMAL
                PMR[x][y] = (objective_value(model), model)
            else
                println(termination_status(model))
            end
        end
    end
    MR = Dict{Pareto,Tuple{Pareto,Tuple{Float64,Model}}}()
    for x in X
        mr = max_regret(PMR, x)
        if !isnothing(mr)
            MR[x] = mr
        end
    end

    if isempty(MR)
        return nothing
    end

    MMR = reduce((x, y) -> x[2][2][1] < y[2][2][1] ? x : y, pairs(MR))

    x_star = MMR[1]
    y_star = MMR[2][1]

    if is_preferred(weight, x_star.objectives, y_star.objectives)
        push!(preferences, (x_star, y_star))
    else
        is_preferred(weight, y_star.objectives, x_star.objectives)
        push!(preferences, (y_star, x_star))
    end

    return MMR
end

# """
# Returns the optimal solution when the decision-maker's preferences are 
# modeled by a weighted sum.

# # Parameters
# - `X::Vector`: A list of all potentially Pareto optimal solutions found with PLS.
# - `weight::Vector`: A vector of weights for the decision-maker.

# # Returns
# - `Tuple`: A tuple in the form of (optimal solution, value of the optimal solution for the decision-maker).
# """
# function find_optimal_solution(X::Vector{Pareto}, weight::Vector{Float64})
#     weighted_sums = [wsum(x.objectives, weight) for x in X]
#     index = argmax(weighted_sums)
#     return X[index], weighted_sums[index]
# end


function optimal_solver_wsum(knapsack::MultiObjectiveKnapsack, weight::Vector{Float64})
    model = Model()
    @variable(model, x[1:knapsack.n], Bin)
    @expression(model, objective_exp, sum(knapsack.items[i].values' * weight * x[i] for i in 1:knapsack.n))
    @objective(model, Max, objective_exp)
    @constraint(model, sum(knapsack.items[i].weight * x[i] for i in 1:knapsack.n) <= knapsack.capacity)
    set_optimizer(model, COPT.Optimizer)

    set_silent(model)
    optimize!(model)
    # print(solution_summary(model))
    result_num = result_count(model)
    results = Pareto[]
    wsums = Float64[]
    for i in 1:result_num
        solution = value.(x; result=i)
        obj = sum(knapsack.items[i].values * solution[i] for i in 1:knapsack.n)
        push!(results, Pareto(solution, obj))
        push!(wsums, objective_value(model; result=i))
    end
    return results, wsums
end

function run_wsum(path::String, filename::String, number_of_known_preferences::Int; MMRlimit::Float64=0.001)
    # 首先，从文件中加载Pareto解决方案
    mkp, X = read_pls_result(path, filename)
    p = length(X[1].objectives)

    time_run = @elapsed solution_eli, num_questions, value_eli, weight = incremental_elicitation(p, X, number_of_known_preferences; MMRlimit=MMRlimit)

    # solution_optimal, weight_sum = find_optimal_solution(X, weight)

    opts, wsums = optimal_solver_wsum(mkp, weight)
    i = argmax(wsums)
    solution_optimal = opts[i]
    weight_sum = wsums[i]

    # 输出结果
    println("Solution incremental elicitation: ", solution_eli.objectives)
    println("Optimal Solution: ", solution_optimal.objectives)
    println("Number of Questions: ", num_questions)
    println("Value of elicitation and OPT: ", value_eli, "\t", weight_sum)
    println("Weight Vector: ", weight)

    is_equal = isequal(solution_eli.solution, solution_optimal.solution)
    println("Is equal: ", is_equal)
    gap = (weight_sum - value_eli) / value_eli
    println("Gap: ", gap)


    # 将结果保存到文件
    save_elicitation_logs("logs_wsum/", "$(mkp.n)_$(p)", solution_eli, solution_optimal, value_eli, weight_sum, weight, num_questions, time_run, number_of_known_preferences)
end

if abspath(PROGRAM_FILE) == @__FILE__
    n = 100
    p = 2
    run_wsum("logs_pls/", "PLS_$(n)_$(p)", 5)
    # for n in 50:10:80
    #     for p in 3:3
    #         # p = 2
    #         run_wsum("logs_pls/", "PLS_$(n)_$(p)", 5)
    #     end
    # end
end