include("PLS.jl")
include("elicitation_weighted.jl")

function regret_base_local_search(knapsack::MultiObjectiveKnapsack, neighbor_list_size::Int, max_iteration::Int, number_of_known_preferences::Int)
    # 初始化解
    archive = init_solutions(1, knapsack)
    current_solution = postorder_traversal(archive)[1]
    iter = 0
    improved = true
    total_num_questions = 0  # 累计问题数量
    value_optimal = 0
    weight = random_weight(knapsack.dimension)

    while iter < max_iteration && improved
        iter += 1
        improved = false
        neighbors = generate_neighbors(current_solution, knapsack, neighbor_list_size)
        push!(neighbors, current_solution)

        # 增量澄清找到最佳解
        solution_optimal_estimated, num_questions, value_optimal, weight = incremental_elicitation(knapsack.dimension, neighbors, number_of_known_preferences, weight=weight)
        total_num_questions += num_questions

        # 检查是否有改进
        if !isequal(solution_optimal_estimated.solution, current_solution.solution)
            current_solution = solution_optimal_estimated
            improved = true
        end
    end

    return current_solution, total_num_questions, value_optimal, weight
end


function main(; dirname="./Data/", filename="2KP200-TA-0", n=40, p=2, list_size=4)
    typename = ".dat"
    mkp = readfile(dirname * filename * typename, n, p)
    number_of_known_preferences = 5
    time_run = @elapsed solution_rbls, num_questions, value_optimal_rbls, weight = regret_base_local_search(mkp, list_size, 100, number_of_known_preferences)

    println("solution_ls: ", solution_rbls.objectives)
    println("num_question: ", num_questions)
    println("value_ls: ", value_optimal_rbls)
    println("weight: ", weight)

    results, wsums = optimal_solver_wsum(mkp, weight)
    i = argmax(wsums)
    sopt = results[i]
    vopt = wsums[i]
    println("solution_optimal: ", sopt.objectives)
    println("value_optimal: ", vopt)

    save_elicitation_logs("logs_rbls/wsum/", "$(n)_$(p)", solution_rbls, sopt, value_optimal_rbls, vopt, weight, num_questions, time_run, number_of_known_preferences)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(n=90, p=2)
end