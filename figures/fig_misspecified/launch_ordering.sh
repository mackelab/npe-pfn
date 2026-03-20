# logging to file
exec 1> >(tee "results/ordering/ordering.log") 2> >(tee "results/ordering/ordering.err" >&2)

task=orderedsimplenonlinear
permutations=(
    [0, 1, 2, 3]  # default/no permutation
    [3, 1, 0, 2]
    [0, 1, 3, 2]
    [0, 2, 1, 3]
    [0, 2, 3, 1]
    [1, 3, 2, 0]
    [2, 0, 3, 1]
    [1, 2, 0, 3]
    [3, 2, 1, 0]
    [2, 1, 3, 0]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=order_tabpfn task=$task "task.params.permutation=$permutation" || true
done

task=orderedmixeddist
permutations=(
    [0, 1, 2]
    [0, 2, 1]
    [1, 0, 2]
    [1, 2, 0]
    [2, 0, 1]
    [2, 1, 0]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=order_tabpfn task=$task "task.params.permutation=$permutation" || true
done

