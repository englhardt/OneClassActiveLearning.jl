struct NoisyOracle <: Oracle
    oracle::Oracle
    p::Float64
end

function NoisyOracle(data, labels, params::Dict{Symbol, Any})
    suboracle = initialize_oracle(eval(params[:subtype]), data, labels, get(params, :subparams, Dict{Symbol, Any}()))
    return NoisyOracle(suboracle, params[:p])
end

function ask_oracle(oracle::NoisyOracle, queries::Union{Vector{Int}, Array{T, 2}})::Vector{Symbol} where T <: Real
    labels = ask_oracle(oracle.oracle, queries)
    flipped_labels = [x == :inlier ? :outlier : :inlier for x in labels]
    mask = rand(length(labels)) .< oracle.p
    labels[mask] .= flipped_labels[mask]
    return labels
end
