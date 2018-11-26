struct RandomOutlierPQs <: ModelBasedPQs
    occ::SVDD.OCClassifier
end

function qs_score(qs::RandomOutlierPQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    prediction = SVDD.classify.(SVDD.predict(qs.occ, x))
    scores = rand(size(x, 2))
    scores[prediction .== :inlier] .= 0
    return scores
end

function qs_score(qs::RandomOutlierPQs,
                  x::Array{T, 2},
                  labels::Dict{Symbol, Array{Int, 1}},
                  subspaces::Vector{Vector{Int}}) where T <: Real
    predictions = map(idx -> SVDD.predict(qs.occ, x[subspaces[idx], :], idx), eachindex(subspaces))
    classifications = SVDD.classify(predictions, Val(:Subspace))
    scores = qs_score(RandomPQs(), x, labels, subspaces)
    for idx in eachindex(subspaces)
        scores[idx][classifications[idx] .== :inlier] .= 0
    end
    return scores
end
