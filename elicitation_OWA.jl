using Random
using LinearAlgebra
using StatsBase
using Combinatorics
using JuMP
using COPT
using Gurobi

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
function max_regret(PMR::Dict, x::Pareto)
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


function elicitation_incrementale_OWA(p::Int, X::Array{Pareto}, nb_pref_connues::Int; MMRlimit::Float64=0.001, decideur=nothing)
    if decideur === nothing
        decideur = random_weight(p)
    end
    sort!(decideur, rev=true)
    println("decideur $decideur")
    if length(X) == 1
        println("Une seule solution possible dans X ! C'est l'optimal")
        valeurOPT = 0
        sort!(X[1])
        for (pi, xi) in zip(decideur, X[1])
            valeurOPT += pi * xi
        end
        return X[1], 0, valeurOPT, decideur, 0
    end
    println("nb_pref_connues = $nb_pref_connues")
    for x in X
        sort!(x.objectives)
    end

    allPairsSolutions = collect(combinations(X, 2))
    solution_init_pref = rand(allPairsSolutions, nb_pref_connues)
    preference = []  # P
    for (x, y) in solution_init_pref
        if is_preferred(decideur, x.objectives, y.objectives)
            push!(preference, (x, y))
        elseif is_preferred(decideur, y.objectives, x.objectives)
            push!(preference, (y, x))
        end
    end


    println("itération n°1")
    MMR = one_question_elicitation_OWA(X, preference, decideur)
    println("Question : \nx : $(MMR[1])\ny : $(MMR[2][1])\nregret : $(MMR[2][2][1])")
    nb_question = 1

    while MMR[2][2][1] > MMRlimit
        println("\nitération n° $(nb_question + 1)\n")
        MMR = one_question_elicitation_OWA(X, preference, decideur)
        println("Question : \nx : $(MMR[1])\ny : $(MMR[2][1])\nregret : $(MMR[2][2][1])")
        nb_question += 1
    end

    valeurOPT = wsum(MMR[1].objectives, decideur)
    println("\nFIN:\nx : $(MMR[1])\ny : $(MMR[2][1])\nregret : $(MMR[2][2][1])\nvaleurOPT : $valeurOPT\nnbQuestion : $nb_question\n")


    println("decideur $decideur")
    return MMR[1], nb_question, valeurOPT, decideur
end

function one_question_elicitation_OWA(X::Array{Pareto}, preference::Array{Any}, decideur::Array{Float64,1})
    p = length(decideur)
    PMR = Dict()

    for x in X
        PMR[x] = Dict()
        for y in X
            if !isequal(x, y)
                model = Model(COPT.Optimizer)
                set_silent(model)
                @variable(model, w[1:p])
                @objective(model, Max, (y.objectives' * w) - (x.objectives' * w))
                @constraint(model, sum(w) == 1)
                for i in 1:(p-1)
                    @constraint(model, w[i] >= w[i+1])
                end

                for (x_pref, y_pref) in preference
                    @constraint(model, (x_pref.objectives' * w) >= (y_pref.objectives' * w))
                end
                for i in 1:p
                    t = zeros(p)
                    t[i] = 1
                    @constraint(model, t' * w <= 1)
                    @constraint(model, t' * w >= 0)
                end

                optimize!(model)

                if termination_status(model) == INFEASIBLE
                    println("Model is infeasible")
                elseif termination_status(model) == OPTIMAL
                    PMR[x][y] = (objective_value(model), model)
                end
            end
        end
    end

    # MR[x] = (y,(regret de prendre x au lieu de y,model))
    MR = Dict()
    for x in X
        mr = max_regret(PMR, x)
        if !isnothing(mr)
            MR[x] = mr
        end
    end

    

    # 找到x,y = pairs(MR)中, y[2][1]最小的
    # @time MMR = sort(collect(MR), by=x -> x[2][2][1])[1]
    # @time MMR = find_min_MR(MR)
    # (x,(y,(regret,model)))
    MMR = reduce((x, y) -> x[2][2][1] < y[2][2][1] ? x : y, pairs(MR))
    
    x_star = MMR[1]
    y_star = MMR[2][1]

    if is_preferred(decideur, x_star.objectives, y_star.objectives)
        push!(preference, (x_star, y_star))
    elseif is_preferred(decideur, y_star.objectives, x_star.objectives)
        push!(preference, (y_star, x_star))
    end

    return MMR
end

function getSolutionOptOWA(X::Array{Array{Float64,1},1}, poids_decideur::Array{Float64,1})
    """
    返回决策者的最优解，这里假设决策者的偏好通过OWA来表示。
    参数:
    - X: 一个数组，包含所有可能的Pareto最优解。
    - poids_decideur: 决策者的权重数组。

    返回值:
    - 一个元组，包含最优解及其对于决策者的价值。
    """
    # 确保每个解都是排序后的
    for x in X
        sort!(x.objectives)
    end

    # 计算每个解的加权总和
    valeursSP = [(x, wsum(x.objectives, poids_decideur)) for x in X]

    # 寻找加权总和最大的解
    return maximum(valeursSP, by=x -> x[2])
end

function test()
    X = Pareto[Pareto([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [7767, 6782]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0], [7965, 5897]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [8030, 5164]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [6771, 6870]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [7406, 6823]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], [7853, 6384])] # 示例解决方案

    @time solution_optimal, num_question, value_optimal, weight = elicitation_incrementale_OWA(2, X, 1)

    println("solution_optimal: ", solution_optimal)
    println("num_question: ", num_question)
    println("value_optimal: ", value_optimal)
    println("weight: ", weight)

end

function process_pareto_solutions(filename::String, number_of_known_preferences::Int, MMRlimit::Float64=0.001)
    # 首先，从文件中加载Pareto解决方案
    X = load_pareto_solutions(filename*".txt")
    p = length(X[1].objectives)

    # 执行增量澄清过程
    solution_optimal, num_questions, value_optimal, weight = elicitation_incrementale_OWA(p, X, number_of_known_preferences)

    # 输出结果
    println("Optimal Solution: ", solution_optimal.solution)
    println("Number of Questions: ", num_questions)
    println("Value of Optimal Solution: ", value_optimal)
    println("Weight Vector: ", weight)

    # 将结果保存到文件
    save_pareto_solution(filename*"_OWA_optimal.txt", solution_optimal, value_optimal, weight)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # test()
    process_pareto_solutions("logs/PLS_results_20_3", 5)
end