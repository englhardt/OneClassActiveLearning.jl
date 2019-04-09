mutable struct NaiveExplorativeMarginQss <: HybridQss
    occ::SVDD.SVDDClassifier
    optimizer::QuerySynthesisOptimizer
    boundary_shift_agg_func::Symbol
    lambda::Float64
    use_penalty::Bool
    eps
    function NaiveExplorativeMarginQss(occ, data; optimizer=nothing, boundary_shift_agg_func=:maximum, lambda=1.0, use_penalty=true)
        !isa(occ.kernel_fct, SquaredExponentialKernel) && throw(ArgumentError("Invalid kernel type $(typeof(occ.kernel_fct)). Expected type is a SquaredExponentialKernel."))
        new(occ, optimizer, boundary_shift_agg_func, lambda, use_penalty, nothing)
    end
end

function qs_score_function(qs::NaiveExplorativeMarginQss, data::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Function where T <: Real
    # Init epsilon if not yet set
    if qs.eps === nothing
        data_target, data_outliers = SVDD.generate_binary_data_for_tuning(data)
        qs.eps = max(0.0, qs.lambda * estimate_boundary_shift_epsilon(qs.occ, data_outliers, agg_func=qs.boundary_shift_agg_func))
    end
    occ_modified = deepcopy(qs.occ)
    occ_modified.R += qs.eps
    oc_scoring(x) = -abs.(SVDD.predict(occ_modified, x))

    # Train penalty binary SVM if outlier labels are available
    if qs.use_penalty && haskey(labels, :Lout)
        labels_binary = fill(:inlier, size(data, 2))
        labels_binary[labels[:Lout]] .= :outlier
        model_binary = LIBSVM.svmtrain(data, labels_binary, gamma=qs.occ.kernel_fct.alpha.value.x, cost=100_000.0)
        penalty(x) = LIBSVM.svmpredict(model_binary, x)[2][1, :]
        return x -> oc_scoring(x) .- max.(zeros(size(x, 2)), -penalty(x))
    end
    return oc_scoring
end
