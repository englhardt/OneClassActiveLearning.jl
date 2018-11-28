using Logging

example_scenario = joinpath(@__DIR__, "example_pool_qs.jl")
if !isempty(ARGS)
    length(ARGS) > 1 && error("Please only supply one scenario.")
    example_scenario = joinpath(@__DIR__, "example_$(ARGS[1]).jl")
    isfile(example_scenario) || error("Cannot find scenario file '$(example_scenario)'.")
end
@info "Running with '$example_scenario'"

using SVDD, OneClassActiveLearning, JSON, JuMP, Ipopt, Random, JLD, Gurobi
Random.seed!(0)

function run_experiment(experiment::Dict)
    Random.seed!(0)
    @info "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])"
    if isfile(experiment[:output_file])
        @warn "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])"
        return nothing
    end

    res = Result(experiment)
    errorfile = joinpath(experiment[:log_dir],"$(gethostname())_$(getpid()).error")
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        @warn "Experiment $(experiment[:hash]) finished with unkown error."
        @warn e
    finally
        if res.status[:exit_code] != :success
            @info "Writing error hash to $errorfile."
            open(errorfile, "a") do f
                print(f, "$(experiment[:hash])\n")
            end
        end
        @info "Writing result to $(experiment[:output_file])."
        OneClassActiveLearning.write_result_to_file(experiment[:output_file], res)
    end
end

@info "Generating experiment settings."
include(example_scenario)
@info "Running experiment."
run_experiment(experiment)
