# The subspace example requires the Gurobi solver
using Gurobi
const DATA_FILE = joinpath(@__DIR__, "dummy.csv")
const NUM_DIMENSIONS, NUM_OBSERVATIONS = size(load_data(DATA_FILE)[1])
const SOLVER = with_optimizer(Gurobi.Optimizer; OutputFlag=0)

experiment = Dict{Symbol, Any}(
    :hash => 1,
    :data_file => DATA_FILE,
    :data_set_name => "Dummy",
    :output_file => joinpath(@__DIR__, "example_subspace_qs.json"),
    :log_dir => joinpath(@__DIR__, "example_subspace_qs.log"),
    :split_strategy_name => "Sf",
    :initial_pool_strategy_name => "Pu",
    :model => Dict(:type => :SubSVDD,
                   :param => Dict{Symbol, Any}(:subspaces => [[1,2], [6,7]]),
                   :init_strategy => SimpleSubspaceStrategy(RuleOfThumbScott(),
                                                            FixedCStrategy(0.1),
                                                            gamma_scope=Val(:Subspace))),
    :query_strategy => Dict(:type => :(SubspaceQs{RandomPQs}),
                           :param => Dict{Symbol, Any}(:subspaces => [[1,2], [6,7]])),
    :split_strategy => OneClassActiveLearning.DataSplits(trues(NUM_OBSERVATIONS)),
    :oracle => :PoolOracle,
    :param => Dict(:num_al_iterations => 10,
                   :solver => Dict(:type => SOLVER.constructor,
                                   :flags => Dict(SOLVER.kwargs)),
                   :initial_pools => fill(:U, NUM_OBSERVATIONS),
                   :adjust_K => true,
                   :initial_pool_resample_version => 1))
