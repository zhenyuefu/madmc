using Random
using LinearAlgebra
using StatsBase
using Combinatorics
using JuMP
using COPT
using Gurobi

include("MOKP.jl")
include("elicitation_weighted.jl")

function incremental_elicitation_OWA(p::Int, X::Array{Pareto}, number_of_known_preferences::Int; MMRlimit::Float64=0.001, weights::Union{Vector{Float64},Nothing}=nothing)
    Xb = deepcopy(X)
    x_dict = Dict{Vector{Int},Pareto}()
    for x in X
        x_dict[x.solution] = x
    end
    if weights === nothing
        weights = random_weight(p)
    end
    sort!(weights, rev=true)
    println("decideur $weights")
    if length(Xb) == 1
        println("Une seule solution possible dans X ! C'est l'optimal")
        valeurOPT = 0
        sort!(Xb[1])
        for (pi, xi) in zip(weights, Xb[1])
            valeurOPT += pi * xi
        end
        return Xb[1], 0, valeurOPT, weights, 0
    end
    println("nb_pref_connues = $number_of_known_preferences")
    for x in Xb
        sort!(x.objectives)
    end

    allPairsSolutions = collect(combinations(Xb, 2))
    solution_init_pref = rand(allPairsSolutions, number_of_known_preferences)
    preference = Tuple{Pareto,Pareto}[]  # P
    for (x, y) in solution_init_pref
        if is_preferred(weights, x.objectives, y.objectives)
            push!(preference, (x, y))
        elseif is_preferred(weights, y.objectives, x.objectives)
            push!(preference, (y, x))
        end
    end


    println("itération n°1")
    MMR = one_question_elicitation_OWA(Xb, preference, weights)
    println("Question : \nx : $(MMR[1].objectives)\ny : $(MMR[2][1].objectives)\nregret : $(MMR[2][2][1])")
    nb_question = 1

    while MMR[2][2][1] > MMRlimit
        println("\nitération n° $(nb_question + 1)\n")
        MMR = one_question_elicitation_OWA(Xb, preference, weights)
        println("Question : \nx : $(MMR[1].objectives)\ny : $(MMR[2][1].objectives)\nregret : $(MMR[2][2][1])")
        nb_question += 1
    end

    valeurOPT = wsum(MMR[1].objectives, weights)
    println("\nFIN:\nx : $(MMR[1])\ny : $(MMR[2][1])\nregret : $(MMR[2][2][1])\nvaleurOPT : $valeurOPT\nnbQuestion : $nb_question\n")


    println("decideur $weights")
    return x_dict[MMR[1].solution], nb_question, valeurOPT, weights
end

function one_question_elicitation_OWA(X::Array{Pareto}, preference::Array{Tuple{Pareto,Pareto}}, decideur::Array{Float64,1})
    p = length(decideur)
    PMR = Dict{Pareto,Dict{Pareto,Tuple{Float64,Model}}}()

    for x in X
        PMR[x] = Dict{Pareto,Tuple{Float64,Model}}()
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
    MR = Dict{Pareto,Tuple{Pareto,Tuple{Float64,Model}}}()
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

function get_solution_opt_OWA(X::Array{Pareto}, weight::Array{Float64,1})
    """
    返回决策者的最优解，这里假设决策者的偏好通过OWA来表示。
    参数:
    - X: 一个数组，包含所有可能的Pareto最优解。
    - poids_decideur: 决策者的权重数组。

    返回值:
    - 一个元组，包含最优解及其对于决策者的价值。
    """
    # 确保每个解都是排序后的
    Xb = deepcopy(X)
    for x in Xb
        sort!(x.objectives)
    end

    weighted_sums = [wsum(x.objectives, weight) for x in Xb]
    index = argmax(weighted_sums)
    return X[index], weighted_sums[index]
end

function test()
    X = Pareto[Pareto([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [7767, 6782]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0], [7965, 5897]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [8030, 5164]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [6771, 6870]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [7406, 6823]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], [7853, 6384])] # 示例解决方案

    @time solution_optimal_estimated, num_question, value_optimal, weight = incremental_elicitation_OWA(2, X, 1)

    println("solution_optimal: ", solution_optimal_estimated)
    println("num_question: ", num_question)
    println("value_optimal: ", value_optimal)
    println("weight: ", weight)

end

function run_owa(path::String, filename::String, number_of_known_preferences::Int; MMRlimit::Float64=0.001)
    # 首先，从文件中加载Pareto解决方案
    mkp, X = read_pls_result(path, filename)
    p = length(X[1].objectives)

    # 执行增量澄清过程
    time_run = @elapsed solution_eli, num_questions, value_optimal, weight = incremental_elicitation_OWA(p, X, number_of_known_preferences, MMRlimit=MMRlimit)

    sopt, vopt = get_solution_opt_OWA(X, weight)

    # 输出结果
    println("Optimal Solution: ", sopt.objectives)
    println("Optimal Solution estimated: ", solution_eli.objectives)
    println("Number of Questions: ", num_questions)
    println("Value of elicitation and optimal: ", value_optimal, "\t", vopt)
    println("Weight Vector: ", weight)

    # 将结果保存到文件
    save_elicitation_logs("logs_owa/", "$(mkp.n)_$(p)", solution_eli, sopt, value_optimal, vopt, weight, num_questions, time_run, number_of_known_preferences)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # test()
    n = 40
    p = 2
    run_owa("logs_pls/", "PLS_$(n)_$(p)", 5)
    # for n in 40:10:100
    #     # for p in 2:3
    #     p = 3
    #     run_owa("logs_pls/", "PLS_$(n)_$(p)", 5)
    #     # end
    # end
end