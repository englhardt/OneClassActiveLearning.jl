"""
Original publication:
A. Ghasemi, H. R. Rabiee, M. Fadaee, M. T. Manzuri,
and M. H. Rohban. Active learning from positive and
unlabeled data. In 2011 IEEE 11th International
Conference on Data Mining Workshops, pages
244–250, Dec 2011.
"""
struct ExpectedMaximumEntropyQs <: DataBasedQs
    p_x::Array{Float64, 1}
    bw_method::Union{String, U} where U <: Real
    ExpectedMaximumEntropyQs(x::Array{T, 2}, bw_method="scott") where T <: Real = new(multi_kde(x, bw_method)(x), bw_method)
end

function qs_score(qs::ExpectedMaximumEntropyQs, x::Array{T, 2}, labels::Dict{Symbol, Array{Int, 1}})::Array{Float64, 1} where T <: Real
    haskey(labels, :Lin) || throw(MissingLabelTypeException(:Lin))
    p_x_inlier = multi_kde(x[:, labels[:Lin]], qs.bw_method)(x)
    a = p_x_inlier ./ qs.p_x
    valid_score(a) = (-a^2 * log(a) + a + (a - 1)^2 * log(1 - a)) / (2 * a)
    # fallback for densities with values > 1
    return [(1 - s) <= 0 ? -Inf : valid_score(s) for s in a]
end
