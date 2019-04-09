struct PoolOracle <: Oracle
    labels::Vector{Symbol}
end

function PoolOracle(data, labels, params::Dict{Symbol, Any})
    return PoolOracle(labels)
end

function ask_oracle(oracle::PoolOracle, query_id::Int)
    return oracle.labels[query_id]
end
