#=
slover:
- Julia version: 1.10.0
- Author: zhenyue
- Date: 2024-01-18
=#

using JuMP
using Gurobi
import MultiObjectiveAlgorithms as MOA
using LinearAlgebra

include("ndtree.jl")

function exact_solver(knapsack::MultiObjectiveKnapsack)
    model = Model()
    @variable(model, x[1:knapsack.n], Bin)
    @expression(model, objective_exp, sum(knapsack.items[i].values * x[i] for i in 1:knapsack.n))
    @objective(model, Max, objective_exp)
    @constraint(model, sum(knapsack.items[i].weight * x[i] for i in 1:knapsack.n) <= knapsack.capacity)
    set_optimizer(model, () -> MOA.Optimizer(Gurobi.Optimizer))
    set_attribute(model, MOA.Algorithm(), MOA.DominguezRios())
    set_attribute(model, MOI.TimeLimitSec(), 20)
    set_silent(model)
    optimize!(model)
    # print(solution_summary(model))
    result_num = result_count(model)
    results = Pareto[]
    for i in 1:result_num
        solution = value.(x; result=i)
        obj = objective_value(model; result=i)
        push!(results, Pareto(solution, obj))
    end
    return results
end

function test()
    knapsack = readfile("./Data/20.dat")
    p = exact_solver(knapsack)
    println(p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end
