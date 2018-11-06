
abstract type Oracle end

struct PoolOracle <: Oracle
    labels::Vector{Symbol}
end

function ask_oracle(oracle::PoolOracle, query_id::Int)
    return oracle.labels[query_id]
end

function initialize_oracle(oracle, labels::Vector{Symbol})::Oracle
    if oracle <: PoolOracle
        return PoolOracle(labels)
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
