
const NATIVE_LABEL_ENCODING = LabelEnc.NativeLabels(["inlier", "outlier"])
const LABEL_ENCODING = LabelEnc.NativeLabels([:inlier,:outlier])
const LEARNING_LABEL_ENCODING = LabelEnc.NativeLabels([:Lin,:Lout])
const SplitType = Union{Val{:train}, Val{:test}, Val{:query}}

function load_data(file_path; header=false, native_label_encoding=NATIVE_LABEL_ENCODING)
    raw_data, _ = header ? readdlm(file_path, ',', header=header) : (readdlm(file_path, ','), nothing)
    data = copy(transpose(float.(raw_data[:, 1:end-1])))
    labels = convert_labels_from_raw(raw_data[:,end])
    return data, labels
end

function convert_labels_from_raw(raw_labels, native_label_encoding=NATIVE_LABEL_ENCODING)
    return convertlabel(LABEL_ENCODING, raw_labels, native_label_encoding)
end

function convert_labels_to_learning(labels, label_encoding=LABEL_ENCODING)
    return convertlabel(LEARNING_LABEL_ENCODING, labels, label_encoding)
end

abstract type SplitStrategy end

struct FullSplitStrat <: SplitStrategy end
struct UnlabeledSplitStrat <: SplitStrategy end
struct UnlabeledAndLabeledInlierSplitStrat <: SplitStrategy end
struct LabeledSplitStrat <: SplitStrategy end
struct LabeledInlierSplitStrat <: SplitStrategy end
struct LabeledOutlierSplitStrat <: SplitStrategy end

struct DataSplits
    train::BitArray
    test::BitArray
    train_strat::SplitStrategy
    test_strat::SplitStrategy
    query_strat::SplitStrategy
    DataSplits(train::BitArray) = DataSplits(train, FullSplitStrat())
    DataSplits(train::BitArray, strat::SplitStrategy) = DataSplits(train, copy(train), strat)
    DataSplits(train::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy) = new(train, copy(train), train_strat, test_strat, train_strat)
    DataSplits(train::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy, query_strat::SplitStrategy) = new(train, copy(train), train_strat, test_strat, query_strat)
    DataSplits(train::BitArray, test::BitArray, strat::SplitStrategy) = new(train, test, strat, strat, strat)
    DataSplits(train::BitArray, test::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy) = new(train, test, train_strat, test_strat, train_strat)
end

get_mask(ds, pools, ::Val{:train}) = calc_mask(ds.train_strat, ds.train, pools)
get_mask(ds, pools, ::Val{:test}) = calc_mask(ds.test_strat, ds.test, pools)
get_mask(ds, pools, ::Val{:query}) = calc_mask(ds.query_strat, ds.train, pools)

function get_local_idx(global_idx::Int, ds, pools, split::T) where T <: SplitType
    mask = get_mask(ds, pools, split)
    mask[global_idx] || error("$(get_val_type(split)) split does not contain observation with global idx $(global_idx).")
    local_idx = count(mask[1:global_idx])
    return local_idx
end

"""
    filter_query_id(query_ids::Vector{Int}, split_strategy, query_pool_labels, ::Val{:train})

    Returns the query ids that also are in the train split.
"""
function filter_query_id(query_ids::Vector{Int}, split_strategy, query_pool_labels, ::Val{:train})
    query_id_mask = calc_mask(split_strategy.train_strat, split_strategy.train[query_ids], query_pool_labels)
    return query_ids[query_id_mask]
end

get_train(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.train_strat, ds.train, data, pools)
get_test(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.test_strat, ds.test, data, pools)
get_query(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.query_strat, ds.train, data, pools)

function select_subset(strat::SplitStrategy, init_mask::BitArray, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real
    mask = calc_mask(strat, init_mask, pools)
    return (data[:, mask], pools[mask], findall(mask))
end

calc_mask(strat::FullSplitStrat, init_mask, pools) = init_mask
calc_mask(strat::UnlabeledSplitStrat, init_mask, pools) = init_mask .& (pools .== :U)
calc_mask(strat::UnlabeledAndLabeledInlierSplitStrat, init_mask, pools) = init_mask .& ((pools .== :U) .| (pools .== :Lin))
calc_mask(strat::LabeledSplitStrat, init_mask, pools) = init_mask .& ((pools .== :Lin) .| (pools .== :Lout))
calc_mask(strat::LabeledInlierSplitStrat, init_mask, pools) = init_mask .& (pools .== :Lin)
calc_mask(strat::LabeledOutlierSplitStrat, init_mask, pools) = init_mask .& (pools .== :Lout)

function get_initial_pools(data, labels, data_splits, initial_pool_strategy; n=20, p=0.1, x=10)
    if initial_pool_strategy ∉ ["Pu", "Pp", "Pn", "Pnin", "Pa"]
        throw(ArgumentError("Unknown initial pools strategy '$(initial_pool_strategy)'."))
    end
    l = fill(:U, size(data, 2))
    if initial_pool_strategy == "Pa"
        label_candidates = findall(data_splits.train .& (labels .== :inlier))
        n = min(size(data, 1) + x, length(label_candidates))
        l[label_candidates[1:n]] .= convert_labels_to_learning(labels[label_candidates[1:n]])
    elseif initial_pool_strategy ∈ ["Pp", "Pn"]
        p = initial_pool_strategy == "Pp" ? p : min(1.0, n / sum(data_splits.train))
        label_candidates = findall(data_splits.train)
        (label_indices, _), _ = stratifiedobs((label_candidates, labels[data_splits.train]), p=p)
        l[label_indices] .= convert_labels_to_learning(labels[label_indices])
    elseif initial_pool_strategy == "Pnin"
        label_candidates = findall(data_splits.train .& (labels .== :inlier))
        label_indices = shuffle(label_candidates)[1:min(length(label_candidates), n)]
        l[label_indices] .= convert_labels_to_learning(labels[label_indices])
    end
    return l
end

function get_splits_and_init_pools(data, labels, split_strategy, initial_pool_strategy; holdout_p=0.2, kwargs...)
    train = trues(size(data, 2))
    if split_strategy == "Sf"
        data_splits = DataSplits(train, FullSplitStrat())
    elseif split_strategy == "Sh"
        (train_indices, train_labels), (test_indices, test_labels) = stratifiedobs((collect(1:length(labels)), labels), p=1-holdout_p)
        train[test_indices] .= false
        data_splits = DataSplits(train, .~train, FullSplitStrat())
    elseif split_strategy == "Si"
        data_splits = DataSplits(train, LabeledInlierSplitStrat(), FullSplitStrat(), FullSplitStrat())
    elseif split_strategy == "Sl"
        data_splits = DataSplits(train, LabeledSplitStrat(), FullSplitStrat(), LabeledSplitStrat())
    elseif split_strategy == "Slf"
        data_splits = DataSplits(train, LabeledSplitStrat(), FullSplitStrat(), FullSplitStrat())
    else
        throw(ArgumentError("Unknown split strategy '$(split_strategy)'."))
    end
    initial_pools = get_initial_pools(data, labels, data_splits, initial_pool_strategy; kwargs...)
    return data_splits, initial_pools
end
