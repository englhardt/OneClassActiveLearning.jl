mutable struct ExplorativeMarginQss <: HybridQss
    occ::SVDD.SVDDClassifier
    optimizer::QuerySynthesisOptimizer
    lambda::Float64
    eps
    function ExplorativeMarginQss(occ, data; optimizer=nothing, lambda=1.0)
        !isa(occ.kernel_fct, SquaredExponentialKernel) && throw(ArgumentError("Invalid kernel type $(typeof(occ.kernel_fct))."))
        new(occ, optimizer, lambda, nothing)
    end
end

function qs_score_function(qs::ExplorativeMarginQss, data::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Function where T <: Real
    # Init epsilon if not yet set
    if qs.eps === nothing
        data_target, data_outliers = SVDD.generate_binary_data_for_tuning(data)
        qs.eps = qs.lambda * maximum(SVDD.predict(qs.occ, data_outliers))
    end
    occ_modified = deepcopy(qs.occ)
    occ_modified.R += qs.eps
    oc_scoring(x) = -abs.(SVDD.predict(occ_modified, x))

    # Train penalty binary SVM if outlier labels are available
    if haskey(labels, :Lout)
        labels_binary = fill(:inlier, size(data, 2))
        labels_binary[labels[:Lout]] .= :outlier
        model_binary = LIBSVM.svmtrain(data, labels_binary, gamma=qs.occ.kernel_fct.alpha.value.x, cost=100_000.0)
        penalty(x) = LIBSVM.svmpredict(model_binary, x)[2][1, :]
        return x -> oc_scoring(x) .- max.(zeros(size(x, 2)), -penalty(x))
    end
    return oc_scoring
end
