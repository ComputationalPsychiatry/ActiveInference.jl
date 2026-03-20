using HDF5
using ActiveInference
using DataFrames
using CSV

########################################################
############### Loading the results data ###############
########################################################

file_path_res = "ActiveInference.jl/test/pymdp_cross_val/cross_val_results/complete_run_data.h5"

#--------------- Loading the complete_run_julia result ----------------
# Loading the julia A matrix
A_julia = array_of_any(4)
for i in 1:4
    A_julia[i] = h5read(file_path_res, "julia_A_cross_$i")
end

# Loading the julia B matrix
B_julia = array_of_any(3)
for i in 1:3
    B_julia[i] = h5read(file_path_res, "julia_B_cross_$i")
end

# Loading the julia D matrix
D_julia = array_of_any(3)
for i in 1:3
    D_julia[i] = h5read(file_path_res, "julia_D_cross_$i")
end

# Loading the julia final posterior over states
qs_julia = array_of_any(3)
for i in 1:3
    qs_julia[i] = h5read(file_path_res, "julia_qs_$i")
end

#--------------- Loading the complete_run_python result ----------------
# Loading the python A matrix
A_python = array_of_any(4)
for i in 1:4
    A_python[i] = h5read(file_path_res, "python_A_cross_$i")
    A_python[i] = permutedims(A_python[i], [4, 3, 2, 1])
end

# Loading the python B matrix
B_python = array_of_any(3)
for i in 1:3
    B_python[i] = h5read(file_path_res, "python_B_cross_$i")
    B_python[i] = permutedims(B_python[i], [3, 2, 1])
end

# Loading the python D matrix
D_python = array_of_any(3)
for i in 1:3
    D_python[i] = h5read(file_path_res, "python_D_cross_$i")
end

# Loading the python final posterior over states
qs_python = array_of_any(3)
for i in 1:3
    qs_python[i] = h5read(file_path_res, "python_qs_$i")
end

############################################################
############### cross-validating the results ###############
############################################################
#------------------ Defining decimal place of agreement function ------------------
function round_arrays(arrays, digits)
    [round.(array, digits=digits) for array in arrays]
end

# Rounding to check A
round_n_A = 15
while round_n_A != 0 && !isequal(A_julia, A_python)

    A_julia = round_arrays(A_julia, round_n_A)
    A_python = round_arrays(A_python, round_n_A)
    round_n_A -= 1
end
round_n_A
is_A_equal = isequal(A_julia, A_python)


# Rounding to check B
round_n_B = 15
while round_n_B != 0 && !isequal(B_julia, B_python)

    B_julia = round_arrays(B_julia, round_n_B)
    B_python = round_arrays(B_python, round_n_B)
    round_n_B -= 1
end
round_n_B
is_B_equal = isequal(B_julia, B_python)

# Rounding to check D
round_n_D = 15
while round_n_D != 0 && !isequal(D_julia, D_python)

    D_julia = round_arrays(D_julia, round_n_D)
    D_python = round_arrays(D_python, round_n_D)
    round_n_D -= 1
end
round_n_D
is_D_equal = isequal(D_julia, D_python)

# Rounding to check qs
round_n_qs = 15
while round_n_qs != 0 && !isequal(qs_julia, qs_python)

    qs_julia = round_arrays(qs_julia, round_n_qs)
    qs_python = round_arrays(qs_python, round_n_qs)
    round_n_qs -= 1
end
round_n_qs
is_qs_equal = isequal(qs_julia, qs_python)

#------------------ Creating a DataFrame to store the results ------------------
results_df = DataFrame(
    parameter = ["A", "B", "D", "qs"],
    equivalence = [is_A_equal, is_B_equal, is_D_equal, is_qs_equal],
    to_decimal_place = [round_n_A, round_n_B, round_n_D, round_n_qs]
)

#------------------ Saving the results ------------------
CSV.write("ActiveInference.jl/test/pymdp_cross_val/cross_val_results/results_comparison.csv", results_df)
