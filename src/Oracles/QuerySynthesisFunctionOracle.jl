struct QuerySynthesisFunctionOracle <: Oracle
    f::Function
end

function ask_oracle(oracle::QuerySynthesisFunctionOracle, query_object)
    return oracle.f(query_object)
end
