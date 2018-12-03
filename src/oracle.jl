
abstract type Oracle end

struct PoolOracle <: Oracle
    labels::Vector{Symbol}
end

function PoolOracle(data, labels, params::Dict{Symbol, Any})
    return PoolOracle(labels)
end

function ask_oracle(oracle::PoolOracle, query_id::Int)
    return oracle.labels[query_id]
end

struct QuerySynthesisFunctionOracle <: Oracle
    f
end

function ask_oracle(oracle::QuerySynthesisFunctionOracle, query_object)
    return oracle.f(query_object)
end

struct QuerySynthesisOCCOracle <: Oracle
    classifier::OCClassifier
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data_file::String, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    data, labels = load_data(data_file)
    return QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver, classifier_params=classifier_params, adjust_K=adjust_K)
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    pools = OneClassActiveLearning.convert_labels_to_learning(labels)
    oracle = instantiate(classifier_type, data, pools, classifier_params)
    initialize!(oracle, init_strategy)
    set_adjust_K!(oracle, true)
    SVDD.fit!(oracle, solver)
    return QuerySynthesisOCCOracle(oracle)
end

function QuerySynthesisOCCOracle(data, labels, params::Dict{Symbol, Any})
    check_oracle_parameter(params, key) = key in keys(params) || throw(ArgumentError("Parameter '$key' missing for QuerySynthesisOCCOracle."))
    check_oracle_parameter(params, :classifier_type)
    check_oracle_parameter(params, :init_strategy)
    check_oracle_parameter(params, :solver)
    additional_params = Dict(k => params[k] for k in [:classifier_params, :adjust_K] if k in keys(params))
    return QuerySynthesisOCCOracle(params[:classifier_type], params[:init_strategy], data, labels, params[:solver], additional_params...)
end

function ask_oracle(oracle::QuerySynthesisOCCOracle, query_object)
    return first(SVDD.classify.(SVDD.predict(oracle.classifier, query_object)))
end

struct QuerySynthesisSVMOracle <: Oracle
    classifier::LIBSVM.SVM
end

function QuerySynthesisSVMOracle(init_strategy, data_file::String)
    data, labels = load_data(data_file)
    return QuerySynthesisSVMOracle(init_strategy, data, labels)
end

function QuerySynthesisSVMOracle(init_strategy, data, labels)
    if !isa(init_strategy.gamma_strategy, FixedGammaStrategy)
        throw(ArgumentError("Invalid gamma strategy type: $(typeof(init_strategy.gamma_strategy))"))
    end
    if !isa(init_strategy.C_strategy, FixedCStrategy)
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

function initialize_oracle(oracle, data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}())::Oracle where T <: Real
    if isa(oracle, DataType) && oracle <: Oracle
        return oracle(data, labels, params)
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
