
@testset "result" begin

    @testset "summarize" begin
        @testset "standard" begin
            res = OneClassActiveLearning.Result(Dict(:hash => 1))
            for i in 0:5
                for e in [:metric]
                    push!(res.al_history, e, i, float(i))
                    if i > 0
                        push!(res.al_history, :query_history, i, i)
                        push!(res.al_history, :query_labels, i, iseven(i) ? [:outlier] : [:inlier])
                    end
                end
            end
            OneClassActiveLearning.al_summarize!(res)
            al_summary = res.al_summary[:metric]
            @test al_summary[:start_quality] == 0.0
            @test al_summary[:end_quality] == 5.0
            @test al_summary[:maximum] == 5.0
            @test al_summary[:ramp_up] == [1.0, 2.0, 3.0, 4.0, 5.0]
            @test al_summary[:quality_range] == 5.0
            @test al_summary[:total_quality_range] == 5.0
            @test al_summary[:average_end_quality] == [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
            @test al_summary[:average_quality_change] == 1.0
            @test al_summary[:learning_stability] == [1.0, 1.0, 1.0, 1.0, 1.0]
            @test al_summary[:average_gain] == 1.0
            @test al_summary[:average_loss] == 0.0
            @test al_summary[:ratio_of_outlier_queries] == 2 / 5
            # Reyes summary statistics
            @test al_summary[:aulc] == 12.5
            @test al_summary[:reyes_paulc] == 12.5
            @test al_summary[:reyes_naulc] == 0.0
            @test al_summary[:reyes_tpr] == 5.0
            @test al_summary[:reyes_tnr] == 0.0
            @test al_summary[:reyes_tp] == 62.5
        end

        @testset "1 query" begin
            res = OneClassActiveLearning.Result(Dict(:hash => 1))
            for i in 0:1
                for e in [:metric]
                    push!(res.al_history, e, i, float(i))
                end
            end
            push!(res.al_history, :query_history, 1, 1)
            push!(res.al_history, :query_labels, 1, [:inlier])
            OneClassActiveLearning.al_summarize!(res)
            al_summary = res.al_summary[:metric]
            @test al_summary[:start_quality] == 0.0
            @test al_summary[:end_quality] == 1.0
            @test al_summary[:maximum] == 1.0
            @test al_summary[:ramp_up] == [1.0]
            @test al_summary[:quality_range] == 1.0
            @test al_summary[:total_quality_range] == 1.0
            @test al_summary[:average_end_quality] == [0.5, 1.0]
            @test al_summary[:average_quality_change] == 1.0
            @test al_summary[:learning_stability] == [1.0]
            @test al_summary[:average_gain] == 1.0
            @test al_summary[:average_loss] == 0.0
            @test al_summary[:ratio_of_outlier_queries] == 0.0
            # Reyes summary statistics
            @test al_summary[:aulc] == 0.5
            @test al_summary[:reyes_paulc] == 0.5
            @test al_summary[:reyes_naulc] == 0.0
            @test al_summary[:reyes_tpr] == 1.0
            @test al_summary[:reyes_tnr] == 0.0
            @test al_summary[:reyes_tp] == 0.5
        end

        @testset "1 query 0 improvement" begin
            res = OneClassActiveLearning.Result(Dict(:hash => 1))
            for i in 0:1
                for e in [:metric]
                    push!(res.al_history, e, i, 0.0)
                end
            end
            push!(res.al_history, :query_history, 1, 1)
            push!(res.al_history, :query_labels, 1, [:inlier])
            OneClassActiveLearning.al_summarize!(res)
            al_summary = res.al_summary[:metric]
            @test al_summary[:start_quality] == 0.0
            @test al_summary[:end_quality] == 0.0
            @test al_summary[:maximum] == 0.0
            @test al_summary[:ramp_up] == [0.0]
            @test al_summary[:quality_range] == 0.0
            @test al_summary[:total_quality_range] == 0.0
            @test al_summary[:average_end_quality] == [0.0, 0.0]
            @test al_summary[:average_quality_change] == 0.0
            @test al_summary[:learning_stability] == [0.0]
            @test al_summary[:average_gain] == 0.0
            @test al_summary[:average_loss] == 0.0
            @test al_summary[:ratio_of_outlier_queries] == 0.0
            # Reyes summary statistics
            @test al_summary[:aulc] == 0.0
            @test al_summary[:reyes_paulc] == 0.0
            @test al_summary[:reyes_naulc] == 0.0
            @test al_summary[:reyes_tpr] == 0.0
            @test al_summary[:reyes_tnr] == 0.0
            @test al_summary[:reyes_tp] == 0.0
        end
    end
end
