struct RandomQss <: QuerySynthesisStrategy
    # [[min_dim1, min_dim2, min_dim3] [max_dim1, max_dim2, max_dim3]]
    # == [min_dim1 min_dim2 min_dim3; max_dim1 max_dim2 max_dim3]
    limits::Array{Float64, 2}
    function RandomQss(; optimizer=nothing, limits=[[0., 0.] [1., 1.]])
        check_limits(limits)
        new(limits)
    end
end

function get_query_object(qs::RandomQss, data::Array{T, 2}, labels::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    return rand(size(qs.limits, 1), 1) .* (qs.limits[:, 2] .- qs.limits[:, 1]) .+ qs.limits[:, 1]
end
