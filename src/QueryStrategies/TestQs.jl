type TestQs <: QueryStrategy end

qs_score(qs::TestQs, data::Array{T, 2}, pools::Dict{Symbol, Vector{Int}}) where T <: Real = collect(1:size(data, 2))
