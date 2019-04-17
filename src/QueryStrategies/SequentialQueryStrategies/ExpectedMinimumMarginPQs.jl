"""
Original publication:
A. Ghasemi, H. R. Rabiee, M. Fadaee, M. T. Manzuri,
and M. H. Rohban. Active learning from positive and
unlabeled data. In 2011 IEEE 11th International
Conference on Data Mining Workshops, pages
244–250, Dec 2011.
"""
struct ExpectedMinimumMarginPQs <: DataBasedPQs
    p_x::Array{Float64, 1}
    bw_method::Union{String, U} where U <: Real
    ExpectedMinimumMarginPQs(x::Array{T, 2}, bw_method="scott") where T <: Real = new(QueryStrategies.multi_kde(x, bw_method)(x), bw_method)
end

function qs_score(qs::ExpectedMinimumMarginPQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    haskey(labels, :Lin) || throw(QueryStrategies.MissingLabelTypeException(:Lin))
    p_x_inlier = QueryStrategies.multi_kde(x[:, labels[:Lin]], qs.bw_method)(x)
    return (p_x_inlier ./ qs.p_x .- 1) .* sign.(0.5 .- p_x_inlier ./ qs.p_x)
end
