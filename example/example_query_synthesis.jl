const DATA_FILE = joinpath(@__DIR__, "dummy.csv")
const NUM_DIMENSIONS, NUM_OBSERVATIONS = size(load_data(DATA_FILE)[1])
const SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
const INIT_STRAT = SimpleCombinedStrategy(RuleOfThumbScott(), BoundedTaxErrorEstimate(0.05, 0.02, 0.98))

experiment = Dict{Symbol, Any}(
    :hash => 1,
    :data_file => DATA_FILE,
    :data_set_name => "Dummy",
    :output_file => joinpath(@__DIR__, "example_query_synthesis.json"),
    :log_dir => joinpath(@__DIR__, "example_query_synthesis.log"),
    :split_strategy_name => "Sf",
    :initial_pool_strategy_name => "Pu",
    :model => Dict(:type => :SVDDneg,
                   :param => Dict{Symbol, Any}(),
                   :init_strategy => INIT_STRAT),
    :query_strategy => Dict(:type => :DecisionBoundaryQss,
                            :param => Dict{Symbol, Any}(
                                :optimizer => ParticleSwarmOptimization()
                            )),
    :split_strategy => OneClassActiveLearning.DataSplits(trues(NUM_OBSERVATIONS)),
    :oracle => QuerySynthesisOCCOracle(SVDDneg, INIT_STRAT, DATA_FILE, SOLVER),
    :param => Dict(:num_al_iterations => 10,
                   :solver => SOLVER,
                   :initial_pools => fill(:U, NUM_OBSERVATIONS),
                   :adjust_K => true,
                   :initial_pool_resample_version => 1))
