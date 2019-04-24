abstract type QuerySynthesisCVWrapperOracle <: Oracle end

function QuerySynthesisCVWrapperOracle(data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}())::Oracle where T <: Real
    :subtype in keys(params) || throw(ArgumentError("Parameter 'subtype' missing for QuerySynthesisCVWrapperOracle."))
    !(params[:subtype] <: QuerySynthesisCVWrapperOracle) || throw(ArgumentError("Subtype $(params[:subtype]) not allowed for QuerySynthesisCVWrapperOracle."))
    gamma_search_range = get(params, :gamma_search_range_oracle, 10.0.^range(-2, stop=2, length=20))
    C = get(params, :C, 1)
    num_folds = get(params, :num_folds, 5)
    metric = eval(get(params, :metric, :matthews_corr))

    data_inliers, data_outliers = SVDD.generate_binary_data_for_tuning(data)
    if isempty(data_inliers)
        data_merged = hcat(data, data_outliers)
        ground_truth = vcat(labels, fill(:outlier, size(data_outliers, 2)))
    else
        data_merged = hcat(data, data_inliers, data_outliers)
        ground_truth = vcat(labels, fill(:inlier, size(data_inliers, 2)),
                            fill(:outlier, size(data_outliers, 2)))
    end

    folds = StratifiedKfold(ground_truth, num_folds)
    best_gamma = 1.0
    best_score = -Inf
    info(getlogger(@__MODULE__), "[ORACLE] Testing $(length(gamma_search_range)) gamma values.")
    for gamma in gamma_search_range
        info(getlogger(@__MODULE__), "[ORACLE] Testing gamma = $gamma.")
        cur_scores = []
        for f in folds
            train_mask = falses(length(ground_truth))
            train_mask[f] .= true
            test_mask = .~train_mask

            params[:init_strategy] = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(gamma)), SVDD.FixedCStrategy(C))
            model = initialize_oracle(params[:subtype], data_merged[:, train_mask], ground_truth[train_mask], params)

            prediction = [ask_oracle(model, data_merged[:, i:i]) for i in findall(test_mask)]
            cm = ConfusionMatrix(prediction, ground_truth[test_mask], pos_class=:inlier, neg_class=:outlier)
            push!(cur_scores, metric(cm))
        end
        score = mean(cur_scores)
        debug(getlogger(@__MODULE__), "[ORACLE] gamma = $gamma, score = $score.")
        if score >= best_score
            info(getlogger(@__MODULE__), "[ORACLE] New best fond with gamma = $gamma and score = $score.")
            best_gamma = gamma
            best_score = score
        end
    end
    info(getlogger(@__MODULE__), "[ORACLE] Final gamma = $best_gamma with score = $best_score.")
    params[:init_strategy] = SVDD.SimpleCombinedStrategy(SVDD.FixedGammaStrategy(MLKernels.GaussianKernel(best_gamma)), SVDD.FixedCStrategy(C))
    model = initialize_oracle(params[:subtype], data_merged, ground_truth, params)
    return model
end
