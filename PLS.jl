#=
PLS:
- Julia version: 1.10.0
- Author: zhenyue
- Date: 2024-01-17
=#
using Random
using LinearAlgebra


include("slover.jl")
include("ndtree.jl")


function R1(lambda::Vector{Float64}, item::Item, current_capacity::Int, capacity::Int)
    value_sum = dot(lambda, item.values)
    remaining_capacity = capacity - current_capacity
    return value_sum / ((item.weight / (remaining_capacity + 1)))
end

function R2(lambda::Vector{Float64}, item::Item)
    return dot(lambda, item.values) / item.weight
end


function greedy_solution(knapsack::MultiObjectiveKnapsack, lambda::Vector{Float64})
    """
    The greedy heuristic to create a solution.
    """
    solution = zeros(Int, knapsack.n)
    current_capacity = 0
    objective_values = zeros(Float64, knapsack.dimension)
    while current_capacity < knapsack.capacity
        max_ratio = -Inf
        max_index = 0
        # 循环剩余的物品
        for i in 1:knapsack.n
            if solution[i] == 0 && current_capacity + knapsack.items[i].weight <= knapsack.capacity
                ratio = R1(lambda, knapsack.items[i], current_capacity, knapsack.capacity)
                if ratio > max_ratio
                    max_ratio = ratio
                    max_index = i
                end
            end
        end

        if max_index == 0
            break
        end

        # 将最大比率的物品放入背包
        solution[max_index] = 1
        current_capacity += knapsack.items[max_index].weight
        objective_values += knapsack.items[max_index].values
    end
    return solution, objective_values
end

function generate_uniform_weights(S::Int)
    """
    Generate uniformly distributed weight sets for biobjective instances.
    """
    return [[i, 1 - i] for i in range(0, stop=1, length=S)]
end


function generate_random_weights(S::Int, dimension::Int)
    """
    Generate random weight sets for multiobjective instances.
    """
    return [normalize(rand(dimension)) for _ in 1:S]
end


function init_solutions(S::Int, knapsack::MultiObjectiveKnapsack)
    archive = NDTreeNode(Pareto[], zeros(knapsack.dimension), zeros(knapsack.dimension), NDTreeNode[], nothing)
    weight_sets = knapsack.dimension == 2 ? generate_uniform_weights(S) : generate_random_weights(S, knapsack.dimension)

    for lambda in weight_sets
        solution, objective_values = greedy_solution(knapsack, lambda)
        update_archive!(archive, Pareto(solution, objective_values))
    end

    return archive
end


function generate_neighbors(p::Pareto, knapsack::MultiObjectiveKnapsack, L::Int)
    # L1: 物品候选移除列表
    L1 = []
    # L2: 物品候选添加列表
    L2 = []
    solution = p.solution

    current_capacity = sum(item.weight for (index, item) in enumerate(knapsack.items) if solution[index] == 1)

    # For biobjective instances, the weight set λ necessary to the computation of these ratios is determined according to the relative performance of the potentially eﬃcient solution x selected, for the diﬀerent objectives, among the population P. That is better the evaluation of the solution x according to an objective is, higher is the value of the weight according to this objective. 
    if knapsack.dimension == 2
        q = rand()
        q = q > 0.5 ? q : 1 - q
        if p.objectives[1] > p.objectives[2]
            lambda = [q, 1 - q]
        else
            lambda = [1 - q, q]
        end
    else
        lambda = normalize(rand(knapsack.dimension))
    end


    # 生成 L1 和 L2
    for i in 1:knapsack.n
        if solution[i] == 1
            push!(L1, (i, R2(lambda, knapsack.items[i])))
        else
            push!(L2, (i, R1(lambda, knapsack.items[i], current_capacity, knapsack.capacity)))
        end
    end

    # 根据 R2 和 R1 排序 L1 和 L2
    sort!(L1, by=x -> x[2])
    sort!(L2, by=x -> x[2], rev=true)

    # 从L1和L2中选取前L个物品
    L1 = [x[1] for x in L1[1:L]]
    L2 = [x[1] for x in L2[1:L]]
    residual_index = [L1; L2]
    residual_items = [knapsack.items[i] for i in residual_index]
    value_l1 = sum(knapsack.items[i].values for i in L1)

    # merge them to create the residual problem, composed of $(L * 2)$ items. The capacities $W$ of the residual problem are equal to $W -\sum_{\substack{i=1 \\ i \notin L 1}}^{n} w^{i} x_{i}$
    residual_capacity = knapsack.capacity - sum(knapsack.items[i].weight * solution[i] for i in 1:knapsack.n if i ∉ L1)
    # 生成新的背包问题
    residual_knapsack = MultiObjectiveKnapsack(2 * L, knapsack.dimension, residual_capacity, residual_items)

    # 生成新的解
    residual_paretos = exact_solver(residual_knapsack)


    # 将新的解合并到原解中, 生成新的邻居
    neighbors = Pareto[]
    for residual_pareto in residual_paretos

        # copy the solution
        new_solution = copy(solution)
        for (new_index, old_index) in enumerate(residual_index)
            new_solution[old_index] = residual_pareto.solution[new_index]
        end

        push!(neighbors, Pareto(new_solution, residual_pareto.objectives + p.objectives - value_l1))
    end


    return neighbors
end

function pls(knapsack::MultiObjectiveKnapsack, init_soluation_num::Int, neighbor_num::Int)
    """
    The pareto local search algorithm.
    """
    archive = init_solutions(init_soluation_num, knapsack)
    P = deepcopy(archive)
    Pa = deepcopy(archive)

end

function test()
    knapsack = readfile("./Data/2KP200-TA-0.dat")
    tree = init_solutions(10, knapsack)
    s = postorder_traversal(tree)
    neighbors = generate_neighbors(s[1], knapsack, 4)
    # check the value in neighbors is correct
    for neighbor in neighbors
        # recal_value = sum(knapsack.items[i].values for i in 1:knapsack.n if neighbor.solution[i] == 1)
        # println(neighbor.solution)
        # println(recal_value)
        # println(neighbor.objectives)
        # @assert neighbor.objectives == recal_value
        update_archive!(tree, neighbor)
    end
    s = postorder_traversal(tree)
    println(length(s))
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end