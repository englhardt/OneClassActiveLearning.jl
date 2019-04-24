abstract type Oracle end

function initialize_oracle end
function ask_oracle end

function initialize_oracle(oracle, data::Array{T, 2}, labels::Vector{Symbol}, params::Dict{Symbol, Any}=Dict{Symbol, Any}())::Oracle where T <: Real
    if isa(oracle, DataType) && oracle <: Oracle
        return oracle(data, labels, params)
    end
    throw(ErrorException("Unknown oracle $(oracle)."))
end
