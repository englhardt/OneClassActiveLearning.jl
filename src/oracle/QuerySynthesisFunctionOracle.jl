
struct QuerySynthesisFunctionOracle <: Oracle
    f
end

function ask_oracle(oracle::QuerySynthesisFunctionOracle, query_object)
    return oracle.f(query_object)
end
