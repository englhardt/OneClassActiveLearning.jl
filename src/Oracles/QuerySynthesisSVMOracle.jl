
struct QuerySynthesisSVMOracle <: Oracle
    classifier::LIBSVM.SVM
end

function QuerySynthesisSVMOracle(init_strategy, data_file::String)
    data, labels = load_data(data_file)
    return QuerySynthesisSVMOracle(init_strategy, data, labels)
end

function QuerySynthesisSVMOracle(init_strategy, data, labels)
    if !isa(init_strategy.gamma_strategy, SVDD.FixedGammaStrategy)
        throw(ArgumentError("Invalid gamma strategy type: $(typeof(init_strategy.gamma_strategy))"))
    end
    if !isa(init_strategy.C_strategy, SVDD.FixedCStrategy)
        throw(ArgumentError("Invalid C strategy type: $(typeof(init_strategy.C_strategy))"))
    end
    oracle = LIBSVM.svmtrain(data, labels; gamma=MLKernels.getvalue(init_strategy.gamma_strategy.kernel.alpha), cost=float(init_strategy.C_strategy.C))
    return QuerySynthesisSVMOracle(oracle)
end

function QuerySynthesisSVMOracle(data_file::String; gamma_search_range_oracle=10.0.^range(-2, stop=2, length=20), C=1, metric=cohens_kappa)
    data, labels = load_data(data_file)
    return QuerySynthesisSVMOracle(data, labels, gamma_search_range_oracle=gamma_search_range_oracle, C=C, metric=metric)
end

function QuerySynthesisSVMOracle(data, labels; gamma_search_range_oracle=10.0.^range(-2, stop=2, length=20), C=1, metric=cohens_kappa)
    best_gamma = 0
    best_score = -Inf
    for gamma in gamma_search_range_oracle
        c = LIBSVM.svmtrain(data, labels; gamma=float(gamma), cost=float(C))
        prediction = LIBSVM.svmpredict(c, data)[1]
        cm = ConfusionMatrix(prediction, labels, pos_class=:inlier, neg_class=:outlier)
        score = metric(cm)
        if score > best_score
            best_gamma = gamma
        end
    end
    return QuerySynthesisSVMOracle(LIBSVM.svmtrain(data, labels, gamma=float(best_gamma), cost=float(C)))
end

function QuerySynthesisSVMOracle(data, labels, params::Dict{Symbol, Any})
    if :init_strategy in keys(params)
        return QuerySynthesisSVMOracle(params[:init_strategy], data, labels)
    else
        additional_params = Dict(k => params[k] for k in [:gamma_search_range_oracle, :C, :metric] if k in keys(params))
        return QuerySynthesisSVMOracle(data, labels; additional_params...)
    end
end

function ask_oracle(oracle::QuerySynthesisSVMOracle, query_object)
    return first(LIBSVM.svmpredict(oracle.classifier, query_object)[1])
end
