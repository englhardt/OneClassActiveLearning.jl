
abstract type Oracle end

include("PoolOracle.jl")
include("QuerySynthesisFunctionOracle.jl")
include("QuerySynthesisKNNOracle.jl")
include("QuerySynthesisOCCOracle.jl")
include("QuerySynthesisSVMOracle.jl")
include("QuerySynthesisCVWrapperOracle.jl")

function initialize_oracle(oracle, data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}())::Oracle where T <: Real
    if isa(oracle, DataType) && oracle <: Oracle
        return oracle(data, labels, params)
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
