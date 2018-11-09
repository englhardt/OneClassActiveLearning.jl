
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

function initialize_oracle(oracle, labels::Vector{Symbol})::Oracle
    if isa(oracle, DataType) && oracle <: PoolOracle
        return PoolOracle(labels)
    elseif isa(oracle, QuerySynthesisFunctionOracle)
        return oracle
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
