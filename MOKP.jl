#=
MOKP:
- Julia version: 1.10.0
- Author: zhenyue
- Date: 2024-01-16
=#

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

function test()
    knapsack = readfile("./Data/2KP200-TA-0.dat")
    println(knapsack)
end

if abspath(PROGRAM_FILE) == @__FILE__
    test()
end