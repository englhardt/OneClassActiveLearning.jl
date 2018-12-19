struct RandomOutlierQss <: ModelBasedQss
    occ::OCClassifier
    epsilon::Union{Float64, Vector{Float64}}
    max_tries::Int
    function RandomOutlierQss(occ; optimizer=nothing, epsilon=0.1, max_tries=100_000)
        check_epsilon(epsilon)
        if max_tries < 1
            throw(ArgumentError("Invalid number of tries $max_tries."))
        end
        new(occ, epsilon, max_tries)
    end
end

function get_query_object(qs::RandomOutlierQss, data::Array{T, 2}, labels::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    if :U ∉ labels && :Lin ∉ labels
        throw(MissingLabelTypeException(:U, :Lin))
    end
    data_minima, data_maxima = extrema_arrays(data[:, labels .!= :Lout])
    for i in 1:qs.max_tries
        query_candidate = rand_in_hypercube(data_minima, data_maxima, qs.epsilon)
        if first(SVDD.classify.(SVDD.predict(qs.occ, query_candidate))) == :outlier || i == qs.max_tries
            return query_candidate
        end
    end
end
