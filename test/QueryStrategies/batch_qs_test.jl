@testset "BatchQueryStrategies" begin
    BQS = OneClassActiveLearning.QueryStrategies.BatchQueryStrategies

    dummy_data, dummy_labels = load_data(TEST_DATA_FILE)

    classifier = SVDD.SVDDneg(dummy_data, dummy_labels)
    init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(1)), FixedCStrategy(0.5))
    initialize!(classifier, init_strategy)

    empty_params = Dict{Symbol, Any}()
    sequential_strategy = Dict{Symbol, Any}(
        :type => :RandomPQs,
        :param => Dict{Symbol, Any}()
    )
    batch_size = 5

    function test_batch_qs(qs_type, params::Dict{Symbol, Any}, classifier, data, labels, candidate_indices)
        @test_throws Union{UndefKeywordError, MethodError, ArgumentError} initialize_qs(qs_type, classifier, data, empty_params)
        qs = initialize_qs(qs_type, classifier, data, Dict(params))
        for candidates in candidate_indices
            batch_indices = select_batch(qs, data, labels, candidates)
            @test length(batch_indices) == min(length(candidates), batch_size)
            @test issubset(batch_indices, Set(candidates))
        end
    end

    @testset "batch_qs" begin
        labels = labelmap(fill(:U, 10))
        candidate_indices = [collect(inds) for inds in [1:3, 1:10]]

        @testset "RandomBatchQs" begin
            qs_type = RandomBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size
            )
            test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
        end

        @testset "TopK" begin
            qs_type = TopKBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy
            )
            test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
        end

        @testset "GappedTopkBatchQs" begin
            qs_type = GappedTopkBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size,
                :m => 0,
                :SequentialStrategy => sequential_strategy
            )
            @test_throws ArgumentError initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, Dict(params..., :m => -1))
            test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
        end

        @testset "ClusterBatchQs" begin
            qs_type = ClusterBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size
            )
            @test_throws UndefKeywordError initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, empty_params)
            test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
        end

        @testset "ClusterTopKBatchQs" begin
            qs_type = ClusterTopKBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy
            )
            qs = initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, params)
            test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
        end

        @testset "EnsembleBatchQs" begin
            qs_type = EnsembleBatchQs
            params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :solver => TEST_SOLVER
            )
            qs = initialize_qs(EnsembleBatchQs, classifier, dummy_data, params)
            labels_ensemble = labelmap(fill(:U, TEST_DATA_NUM_OBSERVATIONS))
            for candidates in candidate_indices
                try
                    batch_indices = select_batch(qs, dummy_data, labels_ensemble, candidates)
                    @test length(batch_indices) == min(length(candidates), batch_size)
                    @test issubset(batch_indices, Set(candidates))
                catch e
                    # This strategy depends on the solver finding a solution
                    # for the submodels.
                    # The standard Ipopt solver does not always find one.
                    # If no solution is found, this strategy simply does not
                    # work and the use of another solver is recommended.
                    # In that case we skip these tests.
                    if e isa ErrorException
                        println(e)
                        println("Solver could not find a solution, skipping test.")
                        continue
                    else
                        throw(e)
                    end
                end
            end
        end

        @testset "weighted sum objective" begin
            qs_types = [IterativeBatchQs, EnumerativeBatchQs]
            params = [Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => rep,
                :diversity => div,
                :λ_inf => 1,
                :λ_rep => 1,
                :λ_div => 1
            ) for rep in [:KDE] for div in [:AngleDiversity, :EuclideanDistance]]
            invalid_rep_params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => :INVALID,
                :diversity => :AngleDiversity,
                :λ_inf => 1,
                :λ_rep => 1,
                :λ_div => 1)
            invalid_div_params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => :KDE,
                :diversity => :INVALID,
                :λ_inf => 1,
                :λ_rep => 1,
                :λ_div => 1)
            for qs_type in qs_types
                @testset "$(qs_type)" begin
                    for invalid_params in [invalid_rep_params, invalid_div_params]
                        @test_throws ArgumentError initialize_qs(qs_type, classifier, dummy_data, invalid_params)
                    end
                    for filled_params in params
                        @testset "$(filled_params)" begin
                            test_batch_qs(qs_type, filled_params, classifier, dummy_data, labels, candidate_indices)
                        end
                    end
                end
            end
        end

        @testset "hierarchical" begin
            qs_types = [FilterHierarchicalBatchQs, EnumFilterHierarchicalBatchQs]
            params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => :KDE,
                :diversity => :EuclideanDistance
            )
            invalid_rep_params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => :INVALID,
                :diversity => :AngleDiversity)
            invalid_div_params = Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => :KDE,
                :diversity => :INVALID)
            for qs_type in qs_types
                @testset "$(qs_type)" begin
                    for invalid_params in [invalid_rep_params, invalid_div_params]
                        @test_throws ArgumentError initialize_qs(qs_type, classifier, dummy_data, invalid_params)
                    end
                    test_batch_qs(qs_type, params, classifier, dummy_data, labels, candidate_indices)
                end
            end
        end

        @testset "FilterSimilarBatchQs" begin
            qs_type = FilterSimilarBatchQs
            params = [Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :diversity => div
            ) for div in [:AngleDiversity, :EuclideanDistance]]
            @test_throws UndefVarError initialize_qs(qs_type, classifier, dummy_data, invalid_div_params)
            for filled_params in params
                @testset "$(filled_params)" begin
                    test_batch_qs(qs_type, filled_params, classifier, dummy_data, labels, candidate_indices)
                end
            end
        end
    end

    @testset "get_query_objects" begin
        qs = RandomBatchQs(k=4)
        @testset "a" begin
            data = rand(2, 6)
            pools = fill(:U, 6)
            pools[5] = :Lin
            indices = [1, 3, 5, 7, 9, 11]
            history = [[7]]
            @test_throws ArgumentError OneClassActiveLearning.get_query_objects(qs, data, fill(:Lin, 5), indices, history)
            @test OneClassActiveLearning.get_query_objects(qs, data, pools, indices, history) == [1, 3, 5, 11]
        end

        @testset "b" begin
            data = rand(2, 5)
            pools = fill(:U, 5)
            pools[5] = :Lin
            indices = [1, 2, 4, 7, 9]
            history = [[7]]
            @test_throws ArgumentError OneClassActiveLearning.get_query_objects(qs, data, fill(:Lin, 5), indices, history)
            @test OneClassActiveLearning.get_query_objects(qs, data, pools, indices, history) == [1, 2, 4]
        end
    end
end
