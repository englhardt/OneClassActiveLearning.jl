
abstract type Oracle end

struct PoolOracle <: Oracle
    labels::Vector{Symbol}
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

function QuerySynthesisOCCOracle(classifier_type, init_strategy, file, solver; classifier_params=Dict{Symbol, Any}(), adjust_K=true)
    data, labels = load_data(file)
    pools = OneClassActiveLearning.convert_labels_to_learning(labels)
    oracle = instantiate(classifier_type, data, pools, classifier_params)
    initialize!(oracle, init_strategy)
    set_adjust_K!(oracle, true)
    SVDD.fit!(oracle, solver)
    return QuerySynthesisOCCOracle(oracle)
end

function ask_oracle(oracle::QuerySynthesisOCCOracle, query_object)
    return first(SVDD.classify.(SVDD.predict(oracle.classifier, query_object)))
end

struct QuerySynthesisSVMOracle <: Oracle
    classifier::LIBSVM.SVM
end

function QuerySynthesisSVMOracle(init_strategy, file)
    if !isa(init_strategy.gamma_strategy, FixedGammaStrategy)
        throw(ArgumentError("Invalid gamma strategy type: $(typeof(init_strategy.gamma_strategy))"))
    end
    if !isa(init_strategy.C_strategy, FixedCStrategy)
        throw(ArgumentError("Invalid C strategy type: $(typeof(init_strategy.C_strategy))"))
    end
    data, labels = load_data(file)
    oracle = LIBSVM.svmtrain(data, labels; gamma=MLKernels.getvalue(init_strategy.gamma_strategy.kernel.alpha), cost=float(init_strategy.C_strategy.C))
    return QuerySynthesisSVMOracle(oracle)
end

function ask_oracle(oracle::QuerySynthesisSVMOracle, query_object)
    return first(LIBSVM.svmpredict(oracle.classifier, query_object)[1])
end

function initialize_oracle(oracle, labels::Vector{Symbol})::Oracle
    if isa(oracle, DataType) && oracle <: PoolOracle
        return PoolOracle(labels)
    elseif isa(oracle, QuerySynthesisFunctionOracle) ||
            isa(oracle, QuerySynthesisOCCOracle) ||
            isa(oracle, QuerySynthesisSVMOracle)
        return oracle
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
