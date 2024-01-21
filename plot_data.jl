using JLD2
using Plots

include("MOKP.jl")

function read_result(file::String)
    data = load(file)
    # is_equal=is_equal, gap=gap, num_questions=num_questions, run_time=run_time, vopt_estimated=vopt_estimated, value_optimal=value_optimal, sopt_estimated=sopt_estimated, sopt=sopt, weight=weight)
    return data
end

function plot_rbls()
    path = "logs_rbls/wsum/"
    time_runs = Matrix{Float64}(undef, 9, 3)
    gaps = Matrix{Float64}(undef, 9, 3)
    num_questions = Matrix{Int}(undef, 9, 3)
    for n in 20:10:100
        for p in 2:4
            f = path * "$(n)_$(p).jld2"
            data = read_result(f)
            time_runs[n÷10-1, p-1] = data["run_time"]
            if data["is_equal"]
                gaps[n÷10-1, p-1] = 0
            else
                gaps[n÷10-1, p-1] = data["gap"]
            end
            num_questions[n÷10-1, p-1] = data["num_questions"]
        end
    end


    # plot time
    plot(20:10:100, time_runs, label=["p=2" "p=3" "p=4"], xlabel="n", ylabel="time", title="Time of RBLS", lw=2, size=(600, 400), legend=:topleft)
    savefig("time_rbls.png")
    # plot gap
    plot(20:10:100, gaps, label=["p=2" "p=3" "p=4"], xlabel="n", ylabel="gap", title="Gap of RBLS", lw=2, size=(600, 400), legend=:topleft)
    savefig("gap_rbls.png")
    # plot num_questions
    plot(20:10:100, num_questions, label=["p=2" "p=3" "p=4"], xlabel="n", ylabel="num_questions", title="Number of Questions of RBLS", lw=2, size=(600, 400), legend=:topleft)
    savefig("num_questions_rbls.png")
end

plot_rbls()