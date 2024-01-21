#=
MOKP:
- Julia version: 1.10.0
- Author: zhenyue
- Date: 2024-01-16
=#
using JLD2
using Dates

# 定义一个结构来存储每个物品的信息
struct Item
    weight::Int
    values::Vector{Int}
end

# 定义一个结构来存储背包问题的所有数据
struct MultiObjectiveKnapsack
    n::Int # 物品数量
    dimension::Int # 目标维度
    capacity::Int
    items::Vector{Item}
end

struct Pareto
    solution::Vector{Int}
    objectives::Vector{Int}
end

struct Capacity
    capacity::Dict{Set{Int},Float64}
    p::Int

    Capacity(p) = new(Dict{Set{Int},Float64}(), p)
end
# 读取文件并解析数据
function readfile(filename::String)
    n = 0
    dimension = 0
    capacity = 0
    items = Item[]

    open(filename, "r") do file
        for line in eachline(file)
            if startswith(line, "c w")
                tokens = split(line)
                # dimension 是tokens的长度减去前两个元素
                dimension = length(tokens) - 2
            elseif startswith(line, "c") || isempty(line)
                continue  # 注释或空行，跳过
            elseif startswith(line, "n ")
                n = parse(Int, split(line)[2])
            elseif startswith(line, "i ")
                # 解析物品数据
                tokens = split(line)
                weight = parse(Int, tokens[2])
                values = parse.(Int, tokens[3:end])
                push!(items, Item(weight, values))
            elseif startswith(line, "W ")
                # 解析背包容量
                capacity = parse(Int, split(line)[2])
            end
        end
    end

    return MultiObjectiveKnapsack(n, dimension, capacity, items)
end

function readfile(filename::String, n_items::Int, p_objectives::Int)
    items = Item[]
    total_weight = 0  # Used to calculate the sum of weights

    open(filename, "r") do file
        for line in eachline(file)
            if startswith(line, "c w")
                continue  # Skip the line with weights
            elseif startswith(line, "c") || isempty(line)
                continue  # Skip comments or empty lines
            elseif startswith(line, "n ")
                continue  # Skip the total number of items
            elseif startswith(line, "i ") && length(items) < n_items
                tokens = split(line)
                weight = parse(Int, tokens[2])
                values = parse.(Int, tokens[3:end])
                push!(items, Item(weight, values[1:p_objectives]))
                total_weight += weight  # Add the item's weight to the total
            end
        end
    end

    # Set the capacity to half the total weight of the first n_items
    capacity = div(total_weight, 2)

    # Construct the MultiObjectiveKnapsack with the calculated capacity
    return MultiObjectiveKnapsack(n_items, p_objectives, capacity, items)
end

function save_pls_logs(path::String, filename_base::String, mkp::MultiObjectiveKnapsack, solutions::Vector{Pareto}, run_time::Float64)
    # 检查文件夹是否存在
    if !isdir(path)
        mkdir(path)
    end
    # 检查文件是否已经存在
    f = path * filename_base * ".log"
    fj = path * filename_base * ".jld2"
    if isfile(f)
        # 如果文件已经存在，则在文件名后面添加时间戳
        t = string(now())
        f = path * filename_base * "_" * t * ".log"
        fj = path * filename_base * "_" * t * ".jld2"
    end
    open(f, "w") do file
        write(file, "Number of items: $(mkp.n)\n")
        write(file, "Number of objectives: $(mkp.dimension)\n")
        write(file, "Capacity: $(mkp.capacity)\n")
        write(file, "Run time: $(run_time) s\n")
        write(file, "Number of solutions: $(length(solutions))\n")
        write(file, "Solutions:\n")
        for pareto in solutions
            write(file, "$(join(pareto.solution, ",")) | $(join(pareto.objectives, ","))\n")
        end
    end
    jldsave(fj; solutions=solutions, run_time=run_time, n=mkp.n, p=mkp.dimension, capacity=mkp.capacity, items=mkp.items)
    println("data saved to $fj")
    println("logs saved to $f")
end

function read_pls_result(path::String, filename_base::String)
    f = path * filename_base * ".jld2"
    data = load(f)
    mkp = MultiObjectiveKnapsack(data["n"], data["p"], data["capacity"], data["items"])
    return mkp, data["solutions"]
end

function save_elicitation_logs(path::String, filename_base::String, sopt_estimated::Pareto, sopt::Pareto, vopt_estimated::Float64, value_optimal::Float64, weight::Union{Vector{Float64},Capacity}, num_questions::Int, run_time::Float64, number_of_known_preferences::Int)
    if !isdir(path)
        mkdir(path)
    end

    f = path * filename_base * ".log"
    fj = path * filename_base * ".jld2"

    if isfile(f)
        t = string(now())
        f = path * filename_base * "_" * t * ".log"
        fj = path * filename_base * "_" * t * ".jld2"
    end

    is_equal = isequal(sopt_estimated.solution, sopt.solution)
    gap = (value_optimal - vopt_estimated) / value_optimal
    open(f, "w") do file
        write(file, "n p: $filename_base\n")
        write(file, "Number of known preferences: $number_of_known_preferences\n")
        write(file, "Is equal: $is_equal\n")
        write(file, "Gap: $gap\n")
        write(file, "Number of questions: $num_questions\n")
        write(file, "Run time: $run_time s\n")
        write(file, "Value optimal estimated: $vopt_estimated\n")
        write(file, "Value optimal: $value_optimal\n")
        write(file, "Solution optimal estimated: $(join(sopt_estimated.solution, ",")) | $(join(sopt_estimated.objectives, ","))\n")
        write(file, "Solution optimal: $(join(sopt.solution, ",")) | $(join(sopt.objectives, ","))\n")
        write(file, "Weight: $(join(weight, ","))\n")
    end

    jldsave(fj; is_equal=is_equal, gap=gap, num_questions=num_questions, run_time=run_time, vopt_estimated=vopt_estimated, value_optimal=value_optimal, sopt_estimated=sopt_estimated, sopt=sopt, weight=weight)

end

function main()
    knapsack = readfile("./Data/2KP200-TA-0.dat", 20, 3)
    println(knapsack)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end