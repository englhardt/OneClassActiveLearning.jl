struct TestQss <: QuerySynthesisStrategy
    optimizer::QuerySynthesisOptimizer
    TestQss(; optimizer=nothing) = new(optimizer)
end

function qs_score_function(qs::TestQss, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Function where T <: Real
    return x -> -abs.(sum(x, dims=1))
end
