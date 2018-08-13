struct RandomOutlierQs <: ModelBasedQs
    occ::SVDD.OCClassifier
end

function qs_score(qs::RandomOutlierQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    prediction = classify.(predict(qs.occ, x))
    scores = rand(size(x, 2))
    scores[prediction .== :inlier] = 0
    return scores
end
