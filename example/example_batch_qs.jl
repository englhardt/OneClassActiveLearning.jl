const DATA_FILE = joinpath(@__DIR__, "dummy.csv")
const NUM_DIMENSIONS, NUM_OBSERVATIONS = size(load_data(DATA_FILE)[1])
const SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)

experiment = Dict{Symbol, Any}(
    :hash => 1,
    :data_file => DATA_FILE,
    :data_set_name => "Dummy",
    :output_file => joinpath(@__DIR__, "example_batch_qs.json"),
    :log_dir => joinpath(@__DIR__, "example_batch_qs.log"),
    :split_strategy_name => "Sf",
    :initial_pool_strategy_name => "Pu",
    :model => Dict(:type => :SVDDneg,
                   :param => Dict{Symbol, Any}(),
                   :init_strategy => SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))),
    :query_strategy => Dict(:type => :RandomBatchQs,
                            :param => Dict{Symbol, Any}(
                                :k => 8
                            )),
    :split_strategy => OneClassActiveLearning.DataSplits(trues(NUM_OBSERVATIONS)),
    :oracle => Dict{Symbol, Any}(
        :type => :PoolOracle,
        :param => Dict{Symbol, Any}()
    ),
    :param => Dict(:num_al_iterations => 10,
                   :solver => SOLVER,
                   :initial_pools => fill(:U, NUM_OBSERVATIONS),
                   :adjust_K => true,
                   :initial_pool_resample_version => 1))
