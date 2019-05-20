struct QuerySynthesisKNNOracle <: Oracle
    data
    labels::Vector{Symbol}
    k::Int
    dist_func::Function
end

function QuerySynthesisKNNOracle(data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}()) where T <: Real
    k = get(params, :k, 1)
    k % 2 == 0 && throw(ArgumentError("Even value k = $k not allowed in QuerySynthesisKNNOracle to prevent decision ties."))
    dist_func = eval(get(params, :dist_func, :euclidean))
    return QuerySynthesisKNNOracle(data, labels, k, dist_func)
end

function ask_oracle(oracle::QuerySynthesisKNNOracle, query_objects::Array{T, 2})::Vector{Symbol} where T <: Real
    function ask_oracle(q)
        sorted_distances = sort([(i, oracle.dist_func(q, oracle.data[:, i]), oracle.labels[i]) for i in 1:size(oracle.data, 2)], by = x -> x[2])[1:oracle.k]
        return count([x[3] for x in sorted_distances] .== :inlier) > length(sorted_distances) * 0.5 ? :inlier : :outlier
    end
    return [ask_oracle(query_objects[:, i]) for i in 1:size(query_objects, 2)]
end
