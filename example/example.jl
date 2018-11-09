using SVDD, OneClassActiveLearning, JSON, Logging, JuMP, Ipopt, Random

function run_experiment(experiment::Dict)
    Random.seed!(0)
    @info "host: $(gethostname()) worker: $(getpid()) exp_hash: $(experiment[:hash])"
    if isfile(experiment[:output_file])
        @warn "Aborting experiment because the output file already exists. Filename: $(experiment[:output_file])"
        return nothing
    end

    res = Result(experiment)
    errorfile = "$(experiment[:log_dir])worker/$(gethostname())_$(getpid())"
    try
        time_exp = @elapsed res = OneClassActiveLearning.active_learn(experiment)
        res.al_summary[:runtime] = Dict(:time_exp => time_exp)
    catch e
        res.status[:exit_code] = Symbol(typeof(e))
        @warn "Experiment $(experiment[:hash]) finished with unkown error."
        @warn e
    finally
        if res.status[:exit_code] != :success
            @info "Writing error hash to $errorfile.error."
            open("$errorfile.error", "a") do f
                print(f, "$(experiment[:hash])\n")
            end
        end
        @info "Writing result to $(experiment[:output_file])."
        OneClassActiveLearning.write_result_to_file(experiment[:output_file], res)
    end
end

const NUM_OBSERVATIONS = 157

experiment = Dict{Symbol, Any}(
    :hash => 1,
    :data_file => "$(@__DIR__)/dummy.csv",
    :data_set_name => "Dummy",
    :output_file => "$(@__DIR__)/dummy.json",
    :log_dir => "$(@__DIR__)/dummy.log",
    :split_strategy_name => "Sf",
    :initial_pool_strategy_name => "Pu",
    :model => Dict(:type => :SVDDneg,
                   :param => Dict{Symbol, Any}(),
                   :init_strategy => SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))),
    :query_strategy => Dict(:type => :RandomOutlierPQs,
                            :param => Dict{Symbol, Any}()),
    :split_strategy => OneClassActiveLearning.DataSplits(trues(NUM_OBSERVATIONS)),
    :oracle => :PoolOracle,
    :param => Dict(:num_al_iterations => 10,
                   :solver => with_optimizer(Ipopt.Optimizer; print_level=0),
                   :initial_pools => fill(:U, NUM_OBSERVATIONS),
                   :adjust_K => true,
                   :initial_pool_resample_version => 1))

run_experiment(experiment)
