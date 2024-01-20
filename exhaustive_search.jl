
include("ndtree.jl")

function generate_combinations(n::Int)
    return [bitstring(i, n) for i in 0:(2^n-1)]
end

function bitstring(num::Int, len::Int)
    return [parse(Int, c) for c in string(num, base=2, pad=len)]
end

function is_feasible(combination::Vector{Int}, knapsack::MultiObjectiveKnapsack)
    total_weight = sum(combination[i] * knapsack.items[i].weight for i in 1:knapsack.n)
    return total_weight <= knapsack.capacity
end

function calculate_objectives(combination::Vector{Int}, knapsack::MultiObjectiveKnapsack)
    objectives = sum(knapsack.items[i].values for i in 1:knapsack.n if combination[i] == 1)
    return objectives
end


function exhaustive_search(knapsack::MultiObjectiveKnapsack)
    combinations = generate_combinations(knapsack.n)
    # remove all 0 
    combinations = combinations[2:end]
    feasible_solutions = [combination for combination in combinations if is_feasible(combination, knapsack)]
    pareto_solutions = NDTreeNode(Pareto[], zeros(Int, knapsack.dimension), zeros(Int, knapsack.dimension), NDTreeNode[], nothing)

    for solution in feasible_solutions
        objectives = calculate_objectives(solution, knapsack)
        update_archive!(pareto_solutions, Pareto(solution, objectives))
    end
    solutions = postorder_traversal(pareto_solutions)
    return solutions
end

function test()
    knapsack = readfile("./Data/20.dat")
    p = exhaustive_search(knapsack)
    println(p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
