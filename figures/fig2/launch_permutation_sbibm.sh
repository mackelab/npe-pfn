# logging to file
exec 1> >(tee "results/ordering/permutation_sbibm.log") 2> >(tee "results/ordering/permutation_sbibm.err" >&2)

task=permuted
task_name=two_moons
permutations=(
    [0, 1]  # default/no permutation
    [1, 0]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=permuted_sbibm_tabpfn task=$task task.params.task_name=$task_name "task.params.permutation=$permutation" || true
done

task_name=bernoulli_glm
permutations=(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # default/no permutation
    [7, 1, 5, 9, 4, 0, 8, 3, 6, 2]
    [2, 6, 3, 5, 9, 1, 8, 0, 4, 7]
    [1, 7, 6, 4, 3, 0, 5, 8, 9, 2]
    [4, 8, 1, 7, 0, 2, 3, 6, 9, 5]
    [6, 9, 3, 7, 5, 4, 2, 1, 8, 0]
    [0, 5, 6, 3, 7, 1, 9, 8, 2, 4]
    [3, 1, 9, 8, 4, 0, 5, 6, 2, 7]
    [5, 0, 7, 6, 8, 3, 9, 1, 2, 4]
    [2, 4, 6, 8, 0, 7, 5, 9, 3, 1]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=permuted_sbibm_tabpfn task=$task task.params.task_name=$task_name "task.params.permutation=$permutation" || true
done

task_name=slcp
permutations=(
    [0, 1, 2, 3, 4] # default/no permutation
    [2, 3, 4, 1, 0]
    [1, 0, 3, 4, 2]
    [3, 2, 1, 0, 4]
    [4, 2, 0, 1, 3]
    [1, 4, 0, 3, 2]
    [0, 2, 4, 1, 3]
    [3, 1, 2, 4, 0]
    [2, 0, 3, 1, 4]
    [4, 3, 1, 0, 2]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=permuted_sbibm_tabpfn task=$task task.params.task_name=$task_name "task.params.permutation=$permutation" || true
done

task_name=sir
permutations=(
    [0, 1] # default/no permutation
    [1, 0]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=permuted_sbibm_tabpfn task=$task task.params.task_name=$task_name "task.params.permutation=$permutation" || true
done

task_name=lotka_volterra
permutations=(
    [0, 1, 2, 3] # default/no permutation
    [3, 0, 2, 1]
    [2, 3, 1, 0]
    [1, 3, 2, 0]
    [1, 2, 3, 0]
    [0, 2, 3, 1]
    [2, 1, 3, 0]
    [3, 1, 0, 2]
    [0, 3, 1, 2]
    [3, 2, 0, 1]
)
for permutation in "${permutations[@]}"; do
    echo "--------------------------------"
    echo "Running task=$task permutation=$permutation..."
    echo "--------------------------------"
    tabpfnbm +experiment=permuted_sbibm_tabpfn task=$task task.params.task_name=$task_name "task.params.permutation=$permutation" || true
done
