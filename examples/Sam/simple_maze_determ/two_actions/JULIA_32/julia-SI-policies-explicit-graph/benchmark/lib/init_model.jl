model = make_model(CONFIG)

make_A(model, CONFIG);
make_B(model, CONFIG);

model.preferences.loc_pref.C[:] = [
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,     # column 1
    -1.0, -1.0, -1.0, -1.0, -0.4, -0.6, -0.7, -0.8, -1.0,     # column 2
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.75, -1.0,    # column 3
    -1.0, -1.0, -1.0,  0.0, -0.2, -0.4, -0.6, -0.7, -1.0,     # column 4
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.65, -1.0,    # column 5
    -1.0, -1.0,  0.4,  0.2,  0.05, -0.15, -0.2, -0.55, -1.0,  # column 6
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.4, -1.0,     # column 7
     1.0,  0.65,  0.4,  0.25,  0.15,  0.0, -0.2, -0.2, -1.0,  # column 8
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0      # column 9
]



start_idx = start_id = findfirst(x -> x == CONFIG.start_cell, model.states.loc.labels)
model.states.loc.D[start_idx] = 1
parameters = AI.get_parameters()
parameters = @set parameters.gamma = 1.0

