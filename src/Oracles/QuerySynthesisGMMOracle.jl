
struct QuerySynthesisGMMOracle <: Oracle
    gmm::GMM
    threshold::Float64
end

function QuerySynthesisGMMOracle(data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}()) where T <: Real
    haskey(params, :file) || throw(ArgumentError("Parameter `:file` missing for loading QuerySynthesisGMMOracle from."))
    f = open(params[:file])
    oracle = deserialize(f)
    close(f)
    isa(oracle, QuerySynthesisGMMOracle) || throw(ErrorException("Unexpected type $(typeof(oracle)) (expected: QuerySynthesisGMMOracle)."))
    return oracle
end

function ask_oracle(oracle::QuerySynthesisGMMOracle, query_object)
    return first(Distributions.pdf(Distributions.MixtureModel(oracle.gmm), query_object)) .< oracle.threshold ? :outlier : :inlier
end
