
@testset "evaluate" begin
    ground_truth = vcat(fill(:outlier,30), fill(:inlier, 20))
    classification = vcat(fill(:outlier,20), fill(:inlier, 10), fill(:outlier,5), fill(:inlier,15))

    positive_class = :outlier
    negative_class = :inlier
    @testset "confusion matrix" begin
        conf_matrix = OneClassActiveLearning.ConfusionMatrix(classification, ground_truth)
        @test conf_matrix.tp == 20
        @test conf_matrix.fp == 5
        @test conf_matrix.fn == 10
        @test conf_matrix.tn == 15
        @test OneClassActiveLearning.get_positive(conf_matrix) == sum(ground_truth .== positive_class)
        @test OneClassActiveLearning.get_negative(conf_matrix) == sum(ground_truth .== negative_class)
        @test OneClassActiveLearning.get_n(conf_matrix) == length(ground_truth)
    end

    @testset "matthews corr" begin
        conf_matrix = OneClassActiveLearning.ConfusionMatrix(2, 3, 4, 5)
        numerator = (2 * 4) - (3 * 5)
        denominator = sqrt((2 + 3) * (2 + 5) * (4 + 3) * (4 + 5))
        expected = numerator/denominator
        @test OneClassActiveLearning.matthews_corr(conf_matrix) ≈ expected
        @test OneClassActiveLearning.matthews_corr(OneClassActiveLearning.ConfusionMatrix(0, 0, 4, 5)) ≈ 0
    end

    @testset "roc auc" begin
        predictions = vcat(rand(10), -rand(20))
        ground_truth = vcat(fill(:outlier, 10), fill(:inlier, 20))

        @test OneClassActiveLearning.roc_auc(predictions, ground_truth) ≈ 1.0
        @test OneClassActiveLearning.roc_auc(predictions, ground_truth, fpr = 0.1, normalize=true) ≈ 1.0
        ground_truth = vcat(fill(:inlier, 10), fill(:outlier, 20))
        @test OneClassActiveLearning.roc_auc(predictions, ground_truth) ≈ 0.0
        ground_truth = vcat(fill(:inlier, 5), fill(:outlier, 25))
        @test OneClassActiveLearning.roc_auc(predictions, ground_truth, fpr = 0.04, normalize=true) .> OneClassActiveLearning.roc_auc(predictions, ground_truth)
    end
end
