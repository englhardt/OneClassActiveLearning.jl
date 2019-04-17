# Query Synthesis Strategies
abstract type QuerySynthesisStrategy <: QueryStrategy end

abstract type DataBasedQss <: QuerySynthesisStrategy end
abstract type ModelBasedQss <: QuerySynthesisStrategy end
abstract type HybridQss <: QuerySynthesisStrategy end

function get_query_object(qs::QuerySynthesisStrategy, query_data::Array{T, 2}, pools::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    return query_synthesis_optimize(qs_score_function(qs, query_data, labelmap(pools)), qs.optimizer, query_data, pools)
end
