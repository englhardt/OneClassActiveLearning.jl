
struct QuerySynthesisOCCOracle <: Oracle
    classifier::SVDD.OCClassifier
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data_file::String, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    data, labels = OneClassActiveLearning.load_data(data_file)
    return QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver, classifier_params=classifier_params, adjust_K=adjust_K)
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    pools = OneClassActiveLearning.convert_labels_to_learning(labels)
    oracle = instantiate(classifier_type, data, pools, classifier_params)
    SVDD.initialize!(oracle, init_strategy)
    SVDD.set_adjust_K!(oracle, true)
    SVDD.fit!(oracle, solver)
    return QuerySynthesisOCCOracle(oracle)
end

function QuerySynthesisOCCOracle(data, labels, params::Dict{Symbol, Any})
    # inline function declaration
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
