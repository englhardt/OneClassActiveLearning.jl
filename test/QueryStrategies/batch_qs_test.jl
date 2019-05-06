@testset "BatchQueryStrategies" begin

    dummy_data, dummy_labels = load_data(TEST_DATA_FILE)

    empty_params = Dict{Symbol, Any}()
    sequential_strategy = Dict{Symbol, Any}(
        :type => :RandomPQs,
        :param => Dict{Symbol, Any}()
    )
    batch_size = 5

    @testset "initialize_qs" begin
        labels = labelmap(fill(:U, 10))
        candidate_indices = [collect(inds) for inds in [1:3, 1:5]]

        @testset "all random" begin
            qs_type = AllRandomBatchQs
            filled_params = Dict{Symbol, Any}(
                :k => batch_size
            )

            @test_throws ArgumentError initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, empty_params)
            qs = initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
            for candidates in candidate_indices
                batch_indices = select_batch(qs, dummy_data, labels, candidates)
                @test length(batch_indices) == min(length(candidates), batch_size)
                @test issubset(batch_indices, Set(candidates))
            end
        end

        @testset "naive extensions" begin
            @testset "TopK" begin
                qs_type = TopKBatchQs
                filled_params = Dict{Symbol, Any}(
                    :k => batch_size,
                    :SequentialStrategy => sequential_strategy
                )
                qs = initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
                for candidates in candidate_indices
                    batch_indices = select_batch(qs, dummy_data, labels, candidates)
                    @test length(batch_indices) == min(length(candidates), batch_size)
                    @test issubset(batch_indices, Set(candidates))
                end
            end # TopK

            @testset "RandomBest" begin
                qs_type = RandomBestBatchQs
                params = [Dict{Symbol, Any}(
                    :k => batch_size,
                    :m => m,
                    :SequentialStrategy => sequential_strategy
                ) for m in [-1, 0]]
                for filled_params in params
                    if filled_params[:m] == -1
                        @test_throws ArgumentError initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
                        continue
                    end
                    qs = initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
                    for candidates in candidate_indices
                        batch_indices = select_batch(qs, dummy_data, labels, candidates)
                        @test length(batch_indices) == min(length(candidates), batch_size)
                        @test issubset(batch_indices, Set(candidates))
                    end
                end
            end # RandomBest
        end # naive extensions

        @testset "region based" begin
            @testset "KMedoids" begin
                qs_type = KMedoidsBatchQs
                filled_params = Dict{Symbol, Any}(
                    :k => batch_size
                )
                @test_throws ArgumentError initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, empty_params)
                qs = initialize_qs(qs_type, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
                for candidates in candidate_indices
                    batch_indices = select_batch(qs, dummy_data, labels, candidates)
                    @test length(batch_indices) == min(length(candidates), batch_size)
                    @test issubset(batch_indices, Set(candidates))
                end
            end # KMedoids

            @testset "Cluster" begin
                filled_params = Dict{Symbol, Any}(
                    :k => batch_size,
                    :SequentialStrategy => sequential_strategy
                )
                qs = initialize_qs(ClusterBatchQs, SVDD.RandomOCClassifier(dummy_data), dummy_data, filled_params)
                for candidates in candidate_indices
                    batch_indices = select_batch(qs, dummy_data, labels, candidates)
                    @test length(batch_indices) == min(length(candidates), batch_size)
                    @test issubset(batch_indices, Set(candidates))
                end
            end # Cluster

            @testset "Ensemble" begin
                filled_params = Dict{Symbol, Any}(
                    :k => batch_size,
                    :SequentialStrategy => sequential_strategy,
                    :solver => TEST_SOLVER
                )
                model = SVDD.SVDDneg(dummy_data, dummy_labels)
                init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(0.1)), FixedCStrategy(0.7))
                initialize!(model, init_strategy)
                qs = initialize_qs(EnsembleBatchQs, model, dummy_data, filled_params)
                labels = labelmap(fill(:U, TEST_DATA_NUM_OBSERVATIONS))
                for candidates in candidate_indices
                    try
                        batch_indices = select_batch(qs, dummy_data, labels, candidates)
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
            end # Ensemgle
        end # region based

        @testset "weighted sum objective" begin
            qs_types = [IterativeBatchQs, IterativeNRBatchQs, EnumerativeBatchQs]
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
            model = VanillaSVDD(dummy_data)
            init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(1)), FixedCStrategy(0.5))
            initialize!(model, init_strategy)
            for qs_type in qs_types
                @testset "$(qs_type)" begin
                    for invalid_params in [invalid_rep_params, invalid_div_params]
                        @test_throws ArgumentError initialize_qs(qs_type, model, dummy_data, invalid_params)
                    end
                    for filled_params in params
                        @testset "$(filled_params)" begin
                            qs = initialize_qs(qs_type, model, dummy_data, filled_params)
                            for candidates in candidate_indices
                                batch_indices = select_batch(qs, dummy_data, labels, candidates)
                                @test length(batch_indices) == min(length(candidates), batch_size)
                                @test issubset(batch_indices, Set(candidates))
                            end
                        end
                    end
                end
            end
        end # weighted sum objective

        @testset "hierarchical" begin
            qs_types = [GreedyHierarchicalBatchQs, EnumHierarchicalBatchQs]
            params = [Dict{Symbol, Any}(
                :k => batch_size,
                :SequentialStrategy => sequential_strategy,
                :representativeness => rep,
                :diversity => div
            ) for rep in [:KDE] for div in [:AngleDiversity, :EuclideanDistance]]
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
            model = VanillaSVDD(dummy_data)
            init_strategy = SimpleCombinedStrategy(FixedGammaStrategy(MLKernels.GaussianKernel(1)), FixedCStrategy(0.5))
            initialize!(model, init_strategy)
            for qs_type in qs_types
                @testset "$(qs_type)" begin
                    for invalid_params in [invalid_rep_params, invalid_div_params]
                        @test_throws ArgumentError initialize_qs(qs_type, model, dummy_data, invalid_params)
                    end
                    for filled_params in params
                        @testset "$(filled_params)" begin
                            qs = initialize_qs(qs_type, model, dummy_data, filled_params)
                            for candidates in candidate_indices
                                batch_indices = select_batch(qs, dummy_data, labels, candidates)
                                @test length(batch_indices) == min(length(candidates), batch_size)
                                @test issubset(batch_indices, Set(candidates))
                            end
                        end
                    end
                end
            end
        end # hierarchical
    end # initialize

    @testset "get_query_objects" begin
        qs = AllRandomBatchQs(k=4)
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
    end # get_query_objects
end # Batch Query Strategies
