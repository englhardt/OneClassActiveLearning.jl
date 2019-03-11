struct RandomQss <: QuerySynthesisStrategy
    epsilon::Union{Float64, Vector{Float64}}
    function RandomQss(; optimizer=nothing, epsilon=0.1)
        check_epsilon(epsilon)
        new(epsilon)
    end
end

function get_query_object(qs::RandomQss, data::Array{T, 2}, labels::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    return rand_in_hyper_rect(extrema_arrays(data[:, labels .!= :Lout])..., qs.epsilon)
end
