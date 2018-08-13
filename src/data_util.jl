
const NATIVE_LABEL_ENCODING = LabelEnc.NativeLabels(["inlier", "outlier"])
const LABEL_ENCODING = LabelEnc.NativeLabels([:inlier,:outlier])
const LEARNING_LABEL_ENCODING = LabelEnc.NativeLabels([:Lin,:Lout])

function load_data(file_path; header=false, native_label_encoding=NATIVE_LABEL_ENCODING)
    raw_data, _ = header ? readdlm(file_path, ',', header=header) : (readdlm(file_path, ','), nothing)
    data = convert(Array{Float64}, raw_data[:, 1:end-1])'
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

type FullSplitStrat <: SplitStrategy end
type UnlabeledSplitStrat <: SplitStrategy end
type UnlabeledAndLabeledInlierSplitStrat <: SplitStrategy end
type LabeledSplitStrat <: SplitStrategy end
type LabeledInlierSplitStrat <: SplitStrategy end
type LabeledOutlierSplitStrat <: SplitStrategy end

struct DataSplits
    train::BitArray
    test::BitArray
    train_strat::SplitStrategy
    test_strat::SplitStrategy
    query_strat::SplitStrategy
    DataSplits(train::BitArray) = DataSplits(train, FullSplitStrat())
    DataSplits(train::BitArray, strat::SplitStrategy) = DataSplits(train, train, strat)
    DataSplits(train::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy) = new(train, train, train_strat, test_strat, train_strat)
    DataSplits(train::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy, query_strat::SplitStrategy) = new(train, train, train_strat, test_strat, query_strat)
    DataSplits(train::BitArray, test::BitArray, strat::SplitStrategy) = new(train, test, strat, strat, strat)
    DataSplits(train::BitArray, test::BitArray, train_strat::SplitStrategy, test_strat::SplitStrategy) = new(train, test, train_strat, test_strat, train_strat)
end

get_train(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.train_strat, ds.train, data, pools)
get_test(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.test_strat, ds.test, data, pools)
get_query(ds::DataSplits, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real = select_subset(ds.query_strat, ds.train, data, pools)

function select_subset(strat::SplitStrategy, init_mask::BitArray, data::Array{T, 2}, pools::Vector{Symbol}) where T <: Real
    mask = calc_mask(strat, init_mask, pools)
    return (data[:, mask], pools[mask], find(mask))
end

calc_mask(strat::FullSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask
calc_mask(strat::UnlabeledSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask .& (pools .== :U)
calc_mask(strat::UnlabeledAndLabeledInlierSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask .& ((pools .== :U) .| (pools .== :Lin))
calc_mask(strat::LabeledSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask .& ((pools .== :Lin) .| (pools .== :Lout))
calc_mask(strat::LabeledInlierSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask .& (pools .== :Lin)
calc_mask(strat::LabeledOutlierSplitStrat, init_mask::BitArray, pools::Vector{Symbol}) = init_mask .& (pools .== :Lout)

function get_initial_pools(data, labels, data_splits, initial_pool_strategy; n=20, p=0.1, x=10)
    if initial_pool_strategy ∉ ["Pu", "Pp", "Pn", "Pa"]
        throw(ArgumentError("Unknown initial pools strategy '$(initial_pool_strategy)'."))
    end
    l = fill(:U, size(data, 2))
    if initial_pool_strategy == "Pa"
        n = min(size(data, 1) + x, size(data, 2))
        l = fill(:U, size(data, 2))
        label_candidates = find(data_splits.train .& (labels .== :inlier))
        l[label_candidates[1:n]] = convert_labels_to_learning(labels[label_candidates[1:n]])
    elseif initial_pool_strategy ∈ ["Pp", "Pn"]
        p = initial_pool_strategy == "Pp" ? p : min(1.0, n / sum(data_splits.train))
        label_candidates = find(data_splits.train)
        (label_indices, _), _ = stratifiedobs((label_candidates, labels[data_splits.train]), p=p)
        l[label_indices] = convert_labels_to_learning(labels[label_indices])
    end
    return l
end

function get_splits_and_init_pools(data, labels, split_strategy, initial_pool_strategy; n=20, p=0.1, x=10)
    train = trues(size(data, 2))
    if split_strategy == "Sf"
        data_splits = DataSplits(train, FullSplitStrat())
    elseif split_strategy == "Sh"
        (train_indices, train_labels), (test_indices, test_labels) = stratifiedobs((collect(1:length(labels)), labels), p=0.8)
        train[test_indices] = false
        data_splits = DataSplits(train, .~train, FullSplitStrat())
    elseif split_strategy == "Si"
        data_splits = DataSplits(train, LabeledInlierSplitStrat(), FullSplitStrat(), FullSplitStrat())
    else
        throw(ArgumentError("Unknown split strategy '$(split_strategy)'."))
    end
    initial_pools = get_initial_pools(data, labels, data_splits, initial_pool_strategy, x=x)
    return data_splits, initial_pools
end
