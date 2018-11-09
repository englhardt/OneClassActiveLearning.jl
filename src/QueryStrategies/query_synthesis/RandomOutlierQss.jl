struct RandomOutlierQss <: ModelBasedQss
    occ::OCClassifier
    limits::Array{Float64, 2}
    max_tries::Int
    function RandomOutlierQss(occ; optimizer=nothing, limits=[[0., 0.] [1., 1.]], max_tries=100_000)
        check_limits(limits)
        if max_tries < 1
            throw(ArgumentError("Invalid number of tries $max_tries."))
        end
        new(occ, limits, max_tries)
    end
end

function get_query_object(qs::RandomOutlierQss, data::Array{T, 2}, labels::Vector{Symbol}, history::Vector{Array{T, 2}})::Array{T, 2} where T <: Real
    for _ in 1:qs.max_tries
        query_candidate = rand(size(qs.limits, 1), 1) .* (qs.limits[:, 2] .- qs.limits[:, 1]) .+ qs.limits[:, 1]
        if first(SVDD.classify.(SVDD.predict(qs.occ, query_candidate))) == :outlier || i == qs.max_tries
            return query_candidate
        end
    end
end
