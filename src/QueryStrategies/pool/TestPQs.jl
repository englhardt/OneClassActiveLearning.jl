struct TestPQs <: PoolQs
  x
  TestPQs(;x = -1) = new(x)
end

qs_score(qs::TestPQs, data::Array{T, 2}, pools::Dict{Symbol, Vector{Int}}) where T <: Real = collect(1:size(data, 2))

function qs_score(qs::TestPQs,
         data::Array{T, 2},
         pools::Dict{Symbol, Vector{Int}},
         subspaces::Vector{Vector{Int}}) where T <: Real
   return [collect(1:size(data, 2)) for i in eachindex(subspaces)]
 end
