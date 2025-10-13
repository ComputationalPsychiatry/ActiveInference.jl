"""
ActionModels - set save history
"""
#=
using ActionModels

function ActionModels.set_save_history!(aif::POMDPActiveInference, save_history::Bool)
    aif.settings.save_history = save_history
end

=#