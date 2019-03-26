
struct QuerySynthesisOCCOracle <: Oracle
    classifier::OCClassifier
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data_file::String, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    data, labels = load_data(data_file)
    return QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver, classifier_params=classifier_params, adjust_K=adjust_K)
end

function QuerySynthesisOCCOracle(classifier_type, init_strategy, data, labels, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    oracle = instantiate(classifier_type, data, labels, classifier_params)
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
