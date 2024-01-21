using Random
using LinearAlgebra
using StatsBase
using JuMP
using COPT
using Combinatorics

include("MOKP.jl")
include("elicitation_weighted.jl")

struct Capacity
    capacity::Dict{Set{Int},Float64}
    p::Int

    Capacity(p) = new(Dict{Set{Int},Float64}(), p)
end

function get_cap(cap::Capacity, x::Set{Int})
    return cap.capacity[x]
end

function set_cap!(cap::Capacity, x, value::Float64)
    cap.capacity[x] = value
end

"""
Function that returns the Choquet value of a solution x with the given "capacite".

# Parameters
- `cap::Capacity`: The capacity for which to calculate the Choquet value.
- `x::Pareto`: A solution to be evaluated.

# Returns
- `Float`: The Choquet value of x with the given capacity "capacite".
"""
function choquet_value(cap::Capacity, x::Pareto)
    choquet_x = x.objectives[1]
    p = length(x.objectives)
    for (i, xi) in enumerate(x.objectives[2:p])
        # println("i: ", i)
        # println("xi: ", xi)
        choquet_x += (xi - x.objectives[i]) * get_cap(cap, Set(range(i + 1, p)))
    end

    return choquet_x
end

function is_domine_choquet(cap::Capacity, x::Pareto, y::Pareto)
    return choquet_value(cap, x) >= choquet_value(cap, y)
end

"""
Returns a completely random convex capacity.

# Parameters
- `p::Int`: The size of the largest set that the capacity will consider.

# Returns
- `capacity`: A randomly generated convex capacity.
"""
function random_convex_capacity(p::Int)
    while true
        cap = Capacity(p)
        set_cap!(cap, Set([]), 0.0)
        values = rand(Float64, 2^p - 2)
        all_combinations = collect(combinations(1:p))
        for (i, A) in enumerate(all_combinations[1:end-1])
            set_cap!(cap, Set(A), values[i])
        end
        set_cap!(cap, Set(all_combinations[end]), 1.0)
        if is_capacity_convex(cap, p)
            return cap
        end
    end
end

function is_capacity_convex(cap::Capacity, p::Int)
    for k in 1:p-1
        A_de_taille_k = collect(combinations(1:p, k))
        for ens in combinations(A_de_taille_k, 2)
            A = collect(ens[1])
            B = collect(ens[2])
            A_and_B = intersect(A, B)
            A_or_B = union(A, B)
            vA_and_B = get_cap(cap, Set(A_and_B))
            vA_or_B = get_cap(cap, Set(A_or_B))
            vA = get_cap(cap, Set(A))
            vB = get_cap(cap, Set(B))
            if vA_and_B + vA_or_B < vA + vB
                return false
            end
        end
    end
    return true
end

"""
Initiates the incremental elicitation of preferences of a randomly chosen decision-maker, 
whose preferences are represented by the Choquet criterion. It's assumed that a number 
of the decision-maker's preferences, "nb_pref_connues," are already known.

# Parameters
- `p::Int`: Number of criteria.
- `X::List`: List of solutions to consider, which are potentially Pareto optimal solutions calculated with PLS.
- `number_of_known_preferences::Int`: Number of known preferences of the decision-maker.
- `MMRlimit::Float`: Optional. A limit at which the elicitation is stopped. This value must be greater than or equal to 0,
  because when MMR < 0, it implies that there exists a solution that will never be regretted for choosing,
  meaning it is the optimal solution. Default is 0.1.

# Returns
- `Tuple`: Estimated optimal solution, the number of questions asked, 
  the value for the decision-maker of the estimated optimal solution, and the weights of the decision-maker.
"""
function incremental_elicitation_choquet(p::Int, X::Array{Pareto}, number_of_known_preferences::Int; MMRlimit=0.1, cap=nothing)
    if isnothing(cap)
        cap = random_convex_capacity(p)
    end
    println("cap: ", cap.capacity)
    if length(X) == 1
        return X[1], 0, choquet_value(cap, X[1]), cap
    end

    for x in X
        sort!(x.objectives)
    end

    all_pairs_of_solutions = collect(combinations(X, 2))
    # random choice number_of_known_preferences pairs of solutions
    known_preferences = sample(all_pairs_of_solutions, number_of_known_preferences, replace=false)
    preferences = []
    for (x, y) in known_preferences
        if is_domine_choquet(cap, x, y)
            push!(preferences, (x, y))
        elseif is_domine_choquet(cap, y, x)
            push!(preferences, (y, x))
        end
    end

    MMR = one_question_elicitation_choquet(X, preferences, cap)
    num_question = 1

    while (MMR[2][2][1] > MMRlimit)
        num_question += 1
        if (num_question >= 50)
            MMRlimit = 1
        end
        if (num_question >= 100)
            MMRlimit = 2
        end
        MMR = one_question_elicitation_choquet(X, preferences, cap)
    end

    value_optimal = choquet_value(cap, MMR[1])

    return MMR[1], num_question, value_optimal, cap
end

function one_question_elicitation_choquet(X::Array{Pareto}, preferences::Array, cap::Capacity)
    p = length(X[1].objectives)
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
            E = collect(combinations(1:p))
            # E union Int[]
            pushfirst!(E, Int[])
           
            vars = Dict{Any,VariableRef}()
            for A in E
                vars[Set(A)] = @variable(model)
            end

            choquet_x = x.objectives[1]
            for (i, xi) in enumerate(x.objectives[2:p])
                choquet_x += (xi - x.objectives[i]) * vars[Set(range(i+1, p))]
            end
            choquet_y = y.objectives[1]
            for (i, yi) in enumerate(y.objectives[2:p])
                choquet_y += (yi - y.objectives[i]) * vars[Set(range(i+1, p))]
            end

            @objective(model, Max, choquet_y - choquet_x)
            for (i, (x_pref, y_pref)) in enumerate(preferences)
                choquet_x_pref = x_pref.objectives[1]
                for (j, xj) in enumerate(x_pref.objectives[2:p])
                    choquet_x_pref += (xj - x_pref.objectives[j]) * vars[Set(range(j+1, p))]
                end
                choquet_y_pref = y_pref.objectives[1]
                for (j, yj) in enumerate(y_pref.objectives[2:p])
                    choquet_y_pref += (yj - y_pref.objectives[j]) * vars[Set(range(j+1, p))]
                end
                @constraint(model, choquet_x_pref >= choquet_y_pref)
            end

            @constraint(model, vars[Set(Int[])] == 0)
            @constraint(model, vars[Set(1:p)] == 1)

            for i in 1:p
                All_A_without_i = collect(combinations(setdiff(1:p, [i])))
                for A in All_A_without_i
                    A_with_i = collect(union(A, [i]))
                    @constraint(model, vars[Set(A)] <= vars[Set(A_with_i)])
                end
            end

            optimize!(model)
            if termination_status(model) == MOI.OPTIMAL
                PMR[x][y] = (objective_value(model), model)
            else
                println(termination_status(model))
            end
        end
    end

    MR = Dict()
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

    if is_domine_choquet(cap, x_star, y_star)
        push!(preferences, (x_star, y_star))
    elseif is_domine_choquet(cap, y_star, x_star)
        push!(preferences, (y_star, x_star))
    end

    return MMR

end

function test()
    X = Pareto[Pareto([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [7767, 6782]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0], [7965, 5897]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], [8030, 5164]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0], [6771, 6870]), Pareto([1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [7406, 6823]), Pareto([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0], [7853, 6384])] # 示例解决方案

    @time solution_optimal, num_question, value_optimal, cap = incremental_elicitation_choquet(2, X, 1)

    println("solution_optimal: ", solution_optimal)
    println("num_question: ", num_question)
    println("value_optimal: ", value_optimal)
    println("cap: ", cap)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end